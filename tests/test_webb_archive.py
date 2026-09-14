import json
from decimal import Decimal
from pathlib import Path

import py7zr
import pyarrow.parquet as pq
import pytest

from hk_quant.webb_archive import read_security_schema, parse_insert_values, SelectedSQLSink, extract_selected_sql, parse_selected_tables

DDL="""CREATE TABLE `issue` (
 `ID1` int unsigned NOT NULL,
 `notes` varchar(255) DEFAULT NULL,
 `flag` bit(1) NOT NULL,
 `price` double DEFAULT NULL,
 `atDate` date DEFAULT NULL,
 PRIMARY KEY (`ID1`)
) ENGINE=InnoDB;
CREATE TABLE `sectypes` (
 `typeID` tinyint unsigned NOT NULL,
 `typeShort` varchar(8) NOT NULL
) ENGINE=InnoDB;
CREATE TABLE `people` (`password` varchar(20)) ENGINE=InnoDB;
"""


def test_actual_mysql_literals_preserve_quotes_commas_null_and_bits():
    sql="INSERT INTO `issue` VALUES (1,'O\\'Brien, A\\\\B\\nC',_binary '\\0',0.027000000000000003,NULL),(2,'it''s (_x_)',_binary '\x01',-2.5e-3,'0000-00-00');"
    rows=list(parse_insert_values(sql,'issue',['ID1','notes','flag','price','atDate']))
    assert rows[0]==[1,"O'Brien, A\\B\nC",b'\x00',Decimal('0.027000000000000003'),None]
    assert rows[1]==[2,"it's (_x_)",b'\x01',Decimal('-.0025'),'0000-00-00']


def test_bit_and_hex_literal_syntax():
    assert list(parse_insert_values("INSERT INTO `t` VALUES (b'101',X'00ff',0xAB);",'t',['a','b','c']))==[[5,b'\x00\xff',b'\xab']]


@pytest.mark.parametrize('sql',[
 'INSERT INTO `issue` VALUES (1,NOW(),NULL);',
 'INSERT INTO `issue` VALUES (1,1+2,NULL);',
 "INSERT INTO `issue` VALUES (1,'unterminated);",
 "INSERT INTO `issue` VALUES (1,'bad\\q',NULL);",
 'INSERT INTO `issue` VALUES (1,NULL); DELETE FROM issue;',
])
def test_nonliteral_or_malformed_sql_fails_without_execution(sql):
    with pytest.raises(ValueError):list(parse_insert_values(sql,'issue',['a','b','c']))


def test_stream_filter_discards_nonwhitelist_bytes_even_across_chunks(tmp_path):
    sink=SelectedSQLSink(tmp_path,{'issue','sectypes'})
    data=(b"-- dump\nINSERT INTO `people` VALUES ('private-login-data');\n"
          b"INSERT INTO `issue` VALUES (1,'public issuer',_binary '\\0',1.2,NULL);\n")
    for i in range(0,len(data),7):sink.write(data[i:i+7])
    sink.close()
    assert [p.name for p in tmp_path.iterdir()]==['issue.sql']
    assert b'private-login-data' not in (tmp_path/'issue.sql').read_bytes()


def test_small_archive_streams_only_data_white_tables_and_parses_types(tmp_path):
    structure=tmp_path/'structure.sql';structure.write_text(DDL)
    data=tmp_path/'source.tmp';data.write_bytes(b"INSERT INTO `people` VALUES ('private');\nINSERT INTO `issue` VALUES (1,'A, B',_binary '\x01',1.25,NULL),(2,NULL,_binary '\\0',NULL,'0000-00-00');\nINSERT INTO `sectypes` VALUES (1,'Ord');\n")
    archive=tmp_path/'sample.7z'
    with py7zr.SevenZipFile(archive,'w') as out:
        out.write(data,'enigmaData-test.sql')
        out.writestr('NEVER EXECUTE', 'enigmaTriggers-test.sql')
    root=tmp_path/'output';schema=read_security_schema(structure)
    assert set(schema)=={'issue','sectypes'}
    extract_selected_sql(archive,root,schema)
    assert not list(root.rglob('enigmaData*')) and not list(root.rglob('enigmaTriggers*'))
    inventory=parse_selected_tables(root,schema)
    table=pq.read_table(root/'parquet/issue.parquet')
    assert table.column_names==['ID1','notes','flag','price','atDate']
    assert table['notes'].to_pylist()==['A, B',None]
    assert table['flag'].to_pylist()==[1,0]
    assert table['atDate'].to_pylist()==[None,'0000-00-00']
    assert inventory['tables']['issue']['rows']==2
    assert inventory['tables']['issue']['null_fraction']['price']==.5
    assert inventory['tables']['issue']['date_ranges']['atDate']['invalid_or_zero_date_values']==1
    assert 'people' not in inventory['tables']


def test_column_count_and_source_table_must_match():
    with pytest.raises(ValueError):list(parse_insert_values('INSERT INTO `issue` VALUES (1,2);','issue',['a']))
    with pytest.raises(ValueError):list(parse_insert_values('INSERT INTO `people` VALUES (1);','issue',['a']))


CCASS_DDL="""CREATE TABLE `quotes` (
 `issueID` mediumint unsigned NOT NULL,
 `atDate` date NOT NULL,
 `prevClose` float unsigned DEFAULT '0',
 `closing` float unsigned NOT NULL DEFAULT '0',
 `ask` float unsigned NOT NULL DEFAULT '0',
 `bid` float unsigned NOT NULL DEFAULT '0',
 `high` float unsigned NOT NULL DEFAULT '0',
 `low` float unsigned NOT NULL DEFAULT '0',
 `vol` bigint unsigned NOT NULL DEFAULT '0',
 `turn` bigint unsigned NOT NULL DEFAULT '0',
 `susp` bit(1) NOT NULL DEFAULT b'0',
 `newsusp` bit(1) NOT NULL DEFAULT b'0',
 `noclose` bit(1) NOT NULL DEFAULT b'0'
) ENGINE=InnoDB;
CREATE TABLE `pquotes` (
 `issueID` mediumint unsigned NOT NULL,
 `atDate` date NOT NULL,
 `prevClose` float unsigned NOT NULL DEFAULT '0',
 `closing` float unsigned NOT NULL DEFAULT '0',
 `ask` float unsigned NOT NULL DEFAULT '0',
 `bid` float unsigned NOT NULL DEFAULT '0',
 `high` float unsigned NOT NULL DEFAULT '0',
 `low` float unsigned NOT NULL DEFAULT '0',
 `vol` bigint unsigned NOT NULL DEFAULT '0',
 `turn` bigint unsigned NOT NULL DEFAULT '0',
 `susp` bit(1) NOT NULL DEFAULT b'0',
 `newsusp` bit(1) NOT NULL DEFAULT b'0',
 `noclose` bit(1) NOT NULL DEFAULT b'0'
) ENGINE=InnoDB;
CREATE TABLE `holdings` (`participant` varchar(20)) ENGINE=InnoDB;
"""


def test_ccass_profile_preserves_actual_quote_fields_and_separate_parallel_table(tmp_path):
    structure=tmp_path/'ccassStructure.sql';structure.write_text(CCASS_DDL+DDL)
    schema=read_security_schema(structure,profile='ccass')
    assert set(schema)=={'quotes','pquotes'}
    payload=(b"INSERT INTO `holdings` VALUES ('participant-data');\n"
             b"INSERT INTO `quotes` VALUES (7,'2010-01-04',NULL,10,10.1,9.9,10.2,9.8,9007199254740993,100000,_binary '\\0',_binary '\x01',_binary '\\0');\n"
             b"INSERT INTO `pquotes` VALUES (7,'2010-01-04',20,20,20.1,19.9,20.2,19.8,100,2000,_binary '\\0',_binary '\\0',_binary '\\0');\n")
    data=tmp_path/'source.bin';data.write_bytes(payload);archive=tmp_path/'ccass.7z'
    with py7zr.SevenZipFile(archive,'w') as writer:writer.write(data,'ccassData-test.sql')
    root=tmp_path/'out';extract_selected_sql(archive,root,schema,profile='ccass')
    inventory=parse_selected_tables(root,schema,profile='ccass')
    quotes=pq.read_table(root/'parquet/quotes.parquet');parallel=pq.read_table(root/'parquet/pquotes.parquet')
    assert quotes['vol'].to_pylist()==[9007199254740993]
    assert quotes['prevClose'].to_pylist()==[None] and quotes['newsusp'].to_pylist()==[1]
    assert quotes['closing'].to_pylist()==[10.] and parallel['closing'].to_pylist()==[20.]
    assert 'open' not in quotes.column_names and 'holdings' not in inventory['tables']
    assert inventory['profile']=='ccass' and set(p.name for p in (root/'selected_sql').iterdir())=={'quotes.sql','pquotes.sql'}


def test_profile_cannot_extract_other_schema_tables(tmp_path):
    structure=tmp_path/'schema.sql';structure.write_text(DDL)
    schema=read_security_schema(structure)
    with pytest.raises(ValueError,match='profile'):
        extract_selected_sql(tmp_path/'not_read.7z',tmp_path/'out',schema,profile='ccass')
