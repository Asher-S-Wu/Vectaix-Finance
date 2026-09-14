"""流式筛选Webb证券表并解析MySQL数据字面量；绝不执行SQL。"""
import argparse
from datetime import datetime,timezone
from decimal import Decimal
import io
import json
import math
import os
from pathlib import Path
import re
import time

import pyarrow as pa
import pyarrow.parquet as pq
import py7zr
from py7zr.io import Py7zIO,WriterFactory

TABLES=('issue','stocklistings','oldlots','hkexdata','entitlements','events','capchangetypes','sectypes','listings','currencies')
PROFILE_TABLES={'enigma':TABLES,'ccass':('quotes','pquotes')}


def _profile_tables(profile):
    if profile not in PROFILE_TABLES:raise ValueError('未知提取profile')
    return PROFILE_TABLES[profile]


def _check_profile(schema,profile):
    if not schema or not {name.lower() for name in schema}.issubset(_profile_tables(profile)):
        raise ValueError('schema超出当前profile白名单或为空')


DEFAULT_ROOT=Path(__file__).resolve().parents[1]/'data/hk/universal/references/webb_archive'
INSERT_BYTES=re.compile(rb'^\s*INSERT\s+INTO\s+(?:`[^`]+`\.)?`([^`]+)`',re.I)
INSERT_TEXT=re.compile(r'^\s*INSERT\s+INTO\s+(?:`[^`]+`\.)?`([^`]+)`',re.I)
BINARY=re.compile(r'_binary\s+',re.I)
HEX_NUMBER=re.compile(r'0x([0-9a-fA-F]+)')
NUMBER=re.compile(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?')


def _save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)


def read_security_schema(path,profile='enigma'):
    allowed=_profile_tables(profile)
    text=Path(path).read_text(encoding='utf-8');schema={}
    for match in re.finditer(r'CREATE TABLE\s+`([^`]+)`\s*\((.*?)\) ENGINE=.*?;',text,re.S):
        name,body=match.groups()
        if name.lower() not in allowed:continue
        columns=[]
        for line in body.splitlines():
            column=re.match(r'\s*`([^`]+)`\s+(.+?)(?:,)?$',line)
            if column:
                field,definition=column.groups()
                kind=re.match(r'(\w+)(?:\(([^)]+)\))?',definition)
                if not kind:raise ValueError(f'{name}.{field}类型无法识别')
                columns.append({'name':field,'mysql_type':kind.group(1).lower(),'type_args':kind.group(2),
                                'unsigned':bool(re.search(r'\bunsigned\b',definition,re.I)),
                                'nullable':'NOT NULL' not in definition.upper(),'definition':definition.rstrip(',')})
        schema[name]=columns
    return schema


class LiteralReader:
    def __init__(self,text,position=0):self.text=text;self.position=position
    def skip(self):
        while self.position<len(self.text) and self.text[self.position].isspace():self.position+=1
    def take(self,character):
        self.skip()
        if not self.text.startswith(character,self.position):raise ValueError(f'字面量语法错误，位置{self.position}应为{character}')
        self.position+=len(character)
    def quoted(self):
        self.take("'");out=[]
        escapes={'0':'\0','b':'\b','n':'\n','r':'\r','t':'\t','Z':'\x1a',"'":"'",'"':'"','\\':'\\','%':'\\%','_':'\\_'}
        while self.position<len(self.text):
            value=self.text[self.position];self.position+=1
            if value=="'":
                if self.text.startswith("'",self.position):out.append("'");self.position+=1;continue
                return ''.join(out)
            if value=='\\':
                if self.position>=len(self.text):raise ValueError('未闭合转义')
                escaped=self.text[self.position];self.position+=1
                if escaped not in escapes:raise ValueError('未支持的MySQL转义序列')
                out.append(escapes[escaped])
            else:out.append(value)
        raise ValueError('未闭合SQL字符串')
    def literal(self):
        self.skip();position=self.position;text=self.text
        if text.startswith("'",position):return self.quoted()
        binary=BINARY.match(text,position)
        if binary:
            self.position=binary.end();return self.quoted().encode('utf-8')
        if position+1<len(text) and text[position] in 'bBxX' and text[position+1]=="'":
            kind=text[position].lower();self.position+=1;raw=self.quoted()
            if kind=='b':
                if not re.fullmatch('[01]+',raw):raise ValueError('BIT字面量无效')
                return int(raw,2)
            if len(raw)%2 or not re.fullmatch('[0-9a-fA-F]*',raw):raise ValueError('HEX字面量无效')
            return bytes.fromhex(raw)
        hexadecimal=HEX_NUMBER.match(text,position)
        if hexadecimal:
            raw=hexadecimal.group(1);self.position=hexadecimal.end()
            return bytes.fromhex(('0' if len(raw)%2 else '')+raw)
        if text[position:position+4].upper()=='NULL':self.position+=4;return None
        number=NUMBER.match(text,position)
        if number:
            token=number.group();self.position=number.end()
            return Decimal(token) if any(c in token.lower() for c in '.e') else int(token)
        raise ValueError(f'只接受SQL数据字面量，位置{self.position}不受支持')


def parse_insert_values(statement,expected_table,expected_columns):
    match=INSERT_TEXT.match(statement)
    if not match or match.group(1)!=expected_table:raise ValueError('INSERT表名或格式不匹配')
    reader=LiteralReader(statement,match.end());reader.skip()
    if statement.startswith('(',reader.position):
        end=statement.find(')',reader.position)
        if end<0:raise ValueError('INSERT列清单未闭合')
        header=statement[reader.position+1:end]
        if not re.fullmatch(r'\s*`[^`]+`(?:\s*,\s*`[^`]+`)*\s*',header):raise ValueError('INSERT列清单包含非法语法')
        names=re.findall(r'`([^`]+)`',header)
        if names!=list(expected_columns):raise ValueError('显式INSERT列清单与DDL顺序不匹配')
        reader.position=end+1
    reader.skip()
    if statement[reader.position:reader.position+6].upper()!='VALUES':raise ValueError('仅接受INSERT VALUES数据')
    reader.position+=6
    while True:
        reader.take('(');row=[]
        while True:
            row.append(reader.literal());reader.skip()
            if statement.startswith(',',reader.position):reader.position+=1;continue
            reader.take(')');break
        if len(row)!=len(expected_columns):raise ValueError('INSERT值数量与DDL列数不一致')
        yield row
        reader.skip()
        if statement.startswith(',',reader.position):reader.position+=1;continue
        reader.take(';');reader.skip()
        if reader.position!=len(statement):raise ValueError('INSERT结束后仍有其他SQL，拒绝解析')
        return


class SelectedSQLSink(Py7zIO):
    def __init__(self,folder,allowed,progress=None):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.allowed=set(allowed);self.progress=progress;self.buffer=b'';self.bytes_seen=0
        self.files={};self.statements={};self.selected_bytes=0;self.closed=False;self.last_update=time.monotonic()
    def _line(self,line):
        match=INSERT_BYTES.match(line)
        if not match:
            if re.match(rb'^\s*INSERT\b',line,re.I):raise ValueError('无法识别INSERT语句头，过滤停止')
            return
        table=match.group(1).decode('ascii')
        if table not in self.allowed:return
        if table not in self.files:
            self.files[table]=(self.folder/(table+'.sql')).open('xb');self.statements[table]=0
        self.files[table].write(line);self.statements[table]+=1;self.selected_bytes+=len(line)
    def write(self,data):
        self.bytes_seen+=len(data);self.buffer+=data
        while True:
            end=self.buffer.find(b'\n')
            if end<0:break
            line=self.buffer[:end+1];self.buffer=self.buffer[end+1:];self._line(line)
        if self.progress and time.monotonic()-self.last_update>=5:
            self.progress(self);self.last_update=time.monotonic()
        return len(data)
    def read(self,size=None):raise io.UnsupportedOperation('过滤器不提供完整Data读取')
    def seek(self,offset,whence=0):
        # py7zr在关闭写入器前调用seek(0)，这里不回退或保存源数据流。
        if offset==0 and whence==0:return 0
        if offset==0 and whence==2:return self.bytes_seen
        raise io.UnsupportedOperation('流式过滤器不支持任意定位')
    def flush(self):
        for file in self.files.values():file.flush()
    def size(self):return self.bytes_seen
    def close(self):
        if self.closed:return
        if self.buffer:self._line(self.buffer);self.buffer=b''
        for file in self.files.values():file.close()
        self.closed=True


class SecuritiesFactory(WriterFactory):
    def __init__(self,sink,profile='enigma'):self.sink=sink;self.profile=profile
    def create(self,filename):
        if not Path(filename).name.startswith(self.profile+'Data-'):raise ValueError('只允许过滤当前profile的Data成员')
        return self.sink


def extract_selected_sql(archive_path,root,schema,profile='enigma'):
    _check_profile(schema,profile)
    archive_path=Path(archive_path);root=Path(root);root.mkdir(parents=True,exist_ok=True)
    status={'status':'extracting','profile':profile,'pid':os.getpid(),'archive_file':str(archive_path.resolve()),
            'archive_bytes':archive_path.stat().st_size,'selected_tables':list(schema),'bytes_seen':0,'selected_bytes':0}
    def update(sink):
        status.update(bytes_seen=sink.bytes_seen,selected_bytes=sink.selected_bytes,statement_counts=dict(sink.statements),updated_at=datetime.now(timezone.utc).isoformat())
        _save(root/'extraction_status.json',status)
    sink=SelectedSQLSink(root/'selected_sql',schema,update)
    _save(root/'extraction_status.json',status)
    try:
        with py7zr.SevenZipFile(archive_path,'r') as archive:
            targets=[n for n in archive.getnames() if Path(n).name.startswith(profile+'Data-')]
            if len(targets)!=1:raise ValueError('档案必须有且只有一个Data成员')
            archive.extract(targets=targets,factory=SecuritiesFactory(sink,profile))
        sink.close();status['status']='finished';update(sink)
    except Exception as exc:
        sink.close();status.update(status='error',error=str(exc));update(sink);raise
    return status


def _arrow_type(column):
    kind=column['mysql_type'];args=column['type_args']
    if kind in ('tinyint','smallint','mediumint','int','integer','bigint'):
        width={'tinyint':8,'smallint':16,'mediumint':32,'int':32,'integer':32,'bigint':64}[kind]
        return getattr(pa,('uint' if column['unsigned'] else 'int')+str(width))()
    if kind=='bit':
        width=int(args)
        if width>64:raise ValueError('BIT宽度超过64位')
        return pa.uint8() if width<=8 else pa.uint64()
    if kind=='float':return pa.float32()
    if kind in ('double','real'):return pa.float64()
    if kind in ('decimal','numeric'):
        precision,scale=map(int,args.split(','));return pa.decimal128(precision,scale)
    if kind in ('date','datetime','timestamp','time','year','char','varchar','text','tinytext','mediumtext','longtext','enum','set'):return pa.string()
    if kind in ('binary','varbinary','blob','tinyblob','mediumblob','longblob'):return pa.binary()
    raise ValueError('不支持的SQL类型: '+kind)


def _convert(value,column):
    if value is None:
        if not column['nullable']:raise ValueError('NOT NULL列出现SQL NULL: '+column['name'])
        return None
    kind=column['mysql_type']
    if kind=='bit':
        if isinstance(value,bytes):value=int.from_bytes(value,'big')
        if not isinstance(value,int) or not 0<=value<2**int(column['type_args']):raise ValueError('BIT值越界')
        return value
    if kind in ('tinyint','smallint','mediumint','int','integer','bigint'):
        if not isinstance(value,(int,Decimal)) or value!=int(value):raise ValueError('整数列不是整数数据字面量')
        return int(value)
    if kind in ('float','double','real'):
        if not isinstance(value,(int,Decimal)):raise ValueError('数值列不是数值字面量')
        number=float(value)
        if not math.isfinite(number):raise ValueError('数值不能表示为有限浮点数')
        return number
    if kind in ('decimal','numeric'):return Decimal(value)
    if pa.types.is_binary(_arrow_type(column)):
        if not isinstance(value,bytes):raise ValueError('二进制列缺少二进制字面量')
        return value
    if not isinstance(value,str):raise ValueError('文本/日期列不是字符串字面量')
    return value


def parse_selected_tables(root,schema,batch_size=5000,provenance=None,profile='enigma'):
    _check_profile(schema,profile)
    root=Path(root)
    extraction=json.loads((root/'extraction_status.json').read_text(encoding='utf-8'))
    if set(extraction['selected_tables'])!=set(schema):raise ValueError('提取表集合与当前profile解析schema不符')
    if extraction['status']!='finished':raise ValueError('白名单流式提取未完成，不解析部分数据')
    output=root/'parquet';output.mkdir(exist_ok=True)
    inventory={'status':'parsing','profile':profile,'pid':os.getpid(),'tables':{},'source_archive':extraction['archive_file'],'source_provenance':provenance,
               'notes':['SQL未执行。仅解析显式数据字面量。','DATE/DATETIME/TIMESTAMP保留原字符串，包括零日期；未推断时区或补齐日期。','数据提取完成不代表公司行动或历史覆盖已经核验。']}
    if profile=='ccass':
        inventory['quote_scope']={'quotes':'normal quoted counter','pquotes':'parallel-trading counter, kept separate',
            'missing_open':'The source DDL has no opening-price field; none is generated.',
            'noclose':'Source flag says closing is not meaningful through 2004-01-30 when suspended all day, or is zero after that date when suspended all day. Raw values and flags remain unchanged.',
            'newsusp':'Source flag includes all-day or part-day suspension in effect at market close.'}
    _save(root/'data_inventory.json',inventory)
    for name,columns in schema.items():
        path=root/'selected_sql'/(name+'.sql');target=output/(name+'.parquet');partial=output/(name+'.parquet.partial')
        arrow_schema=pa.schema([pa.field(c['name'],_arrow_type(c),nullable=c['nullable'],metadata={b'mysql_definition':c['definition'].encode()}) for c in columns])
        stats={'status':'parsing','rows':0,'columns':[dict(c,parquet_type=str(_arrow_type(c))) for c in columns],
               'null_counts':{c['name']:0 for c in columns},'date_ranges':{c['name']:{'first':None,'last':None,'valid_values':0,'invalid_or_zero_date_values':0} for c in columns if c['mysql_type'] in ('date','datetime','timestamp')},
               'source_sql_file':str(path.resolve()) if path.exists() else None,'parquet_file':str(target.resolve())}
        writer=pq.ParquetWriter(partial,arrow_schema,compression='zstd');batch=[];line_number=0;last_update=time.monotonic()
        try:
            if path.exists():
                with path.open('r',encoding='utf-8',newline='') as stream:
                    for line_number,line in enumerate(stream,1):
                        for values in parse_insert_values(line,name,[c['name'] for c in columns]):
                            converted=[_convert(v,c) for v,c in zip(values,columns)]
                            row=dict(zip((c['name'] for c in columns),converted));batch.append(row);stats['rows']+=1
                            for field,value in row.items():
                                if value is None:stats['null_counts'][field]+=1
                                elif field in stats['date_ranges']:
                                    dates=stats['date_ranges'][field]
                                    try:datetime.fromisoformat(value)
                                    except ValueError:dates['invalid_or_zero_date_values']+=1
                                    else:
                                        dates['valid_values']+=1
                                        if dates['first'] is None or value<dates['first']:dates['first']=value
                                        if dates['last'] is None or value>dates['last']:dates['last']=value
                            if len(batch)>=batch_size:
                                writer.write_table(pa.Table.from_pylist(batch,schema=arrow_schema));batch=[]
                                if time.monotonic()-last_update>=5:
                                    inventory['tables'][name]=stats;_save(root/'data_inventory.json',inventory)
                                    print(f'parsing {name}: {stats["rows"]} rows',flush=True);last_update=time.monotonic()
            if batch:writer.write_table(pa.Table.from_pylist(batch,schema=arrow_schema))
            writer.close();partial.replace(target)
            stats['status']='parsed';stats['null_fraction']={field:count/stats['rows'] if stats['rows'] else None for field,count in stats['null_counts'].items()}
            inventory['tables'][name]=stats;_save(root/'data_inventory.json',inventory)
            print(f'parsed {name}: {stats["rows"]} rows',flush=True)
        except Exception as exc:
            writer.close();stats.update(status='error',line=line_number,error=str(exc));inventory['tables'][name]=stats
            inventory['status']='error';_save(root/'data_inventory.json',inventory)
            raise ValueError(f'{name}第{line_number}条INSERT解析失败: {exc}') from exc
    inventory['status']='finished';_save(root/'data_inventory.json',inventory)
    return inventory


def main():
    parser=argparse.ArgumentParser(description='只筛选profile白名单SQL并解析为Parquet，不执行SQL')
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT,help='两个原始7z所在目录')
    parser.add_argument('--profile',choices=['enigma','ccass'],default='enigma')
    parser.add_argument('--stage',choices=['extract','parse','all'],default='all')
    args=parser.parse_args();archive_root=args.root
    root=archive_root/'ccass' if args.profile=='ccass' else archive_root
    archive=archive_root/(args.profile+'251227.7z')
    structures=list(root.glob(args.profile+'Structure-*.sql'))
    if len(structures)!=1:raise ValueError('必须先选择性解出当前profile唯一Structure文件')
    provenance=json.loads((root/'provenance.json').read_text(encoding='utf-8'))
    schema=read_security_schema(structures[0],profile=args.profile)
    _check_profile(schema,args.profile)
    _save(root/'selected_type_dictionary.json',{'source':provenance,'profile':args.profile,'tables':schema})
    print(f'Webb extraction PID={os.getpid()} profile={args.profile} tables={list(schema)} stage={args.stage}',flush=True)
    if args.stage in ('extract','all'):extract_selected_sql(archive,root,schema,profile=args.profile)
    if args.stage in ('parse','all'):parse_selected_tables(root,schema,provenance=provenance,profile=args.profile)


if __name__=='__main__':main()
