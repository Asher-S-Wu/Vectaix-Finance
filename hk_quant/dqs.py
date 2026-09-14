"""读取港交所历史每日报价原件，不填造开盘价、成交或证券身份。"""
import re

from bs4 import BeautifulSoup, NavigableString, Tag
import numpy as np
import pandas as pd


def parse_daily_quotations(document,codes):
    soup=BeautifulSoup(document,'lxml')
    date_match=re.search(r'DATE:\s*(\d{1,2}\s+[A-Z]{3}\s+\d{4})',soup.get_text())
    if not date_match:raise ValueError('报价原件缺少明确日期')
    day=pd.to_datetime(date_match[1],format='%d %b %Y')
    anchor=soup.find('a',attrs={'name':'quotations'})
    if anchor is None:raise ValueError('报价原件缺少QUOTATIONS段落')
    text=[];ended=False
    for element in anchor.next_elements:
        if isinstance(element,Tag) and element.name=='a' and element.get('name')=='sales_all':
            ended=True;break
        if isinstance(element,NavigableString):text.append(str(element))
    if not ended:raise ValueError('报价原件缺少报价段落结束边界')
    content=''.join(text)
    if not all(key in content for key in ('PRV.CLO.','CLOSING','SHARES TRADED','TURNOVER ($)')):
        raise ValueError('报价表头不符合已核验的双行格式')
    requested=set(codes)
    if not all(re.fullmatch(r'\d{5}\.HK',code) for code in requested):raise ValueError('代码格式无效')
    lines=[line for line in content.splitlines() if line.strip()]
    rows=[]
    def number(value):
        if value in ('-','N/A'):return np.nan
        if not re.fullmatch(r'\d+(?:,\d{3})*(?:\.\d+)?',value):raise ValueError('无法识别报价数值: '+value)
        return float(value.replace(',',''))
    for index,line in enumerate(lines):
        head=re.match(r'^\s*(?P<marker>\*)?\s*(?P<code>\d{1,5})(?:(?P<post_marker>#)\s*|\s+)(?P<body>.+)',line)
        if not head:continue
        code=head['code'].zfill(5)+'.HK'
        if code not in requested:continue
        fields=head['body'].split()
        row=dict(date=day,exchange_code=code,quote_marker=head['marker'] or '',code_marker=head['post_marker'] or '',previous_close=np.nan,raw_close=np.nan,raw_high=np.nan,
                 raw_low=np.nan,ask=np.nan,bid=np.nan,volume=np.nan,amount=np.nan,vwap=np.nan)
        if head['body'].endswith('TRADING SUSPENDED'):
            row.update(name=head['body'][:-len('TRADING SUSPENDED')].strip(),status='exchange_reported_suspension',source_lines=line)
        else:
            if len(fields)<5:raise ValueError('报价主行字段不足: '+code)
            if index+1>=len(lines) or re.match(r'^\s*\*?\s*\d{1,5}(?:#\s*|\s+)[A-Za-z]',lines[index+1]):
                raise ValueError('报价缺少双行格式的续行: '+code)
            second=lines[index+1].split()
            if len(second)!=4:raise ValueError('报价续行字段不明确: '+code)
            previous,ask,high,volume=map(number,fields[-4:])
            close,bid,low,amount=map(number,second)
            traded=np.isfinite(volume) and volume>0 and np.isfinite(amount) and amount>0
            if traded and (not all(np.isfinite(v) and v>0 for v in (close,high,low)) or low>high):
                raise ValueError('有成交但高低收价格无效: '+code)
            row.update(name=' '.join(fields[:-4]),previous_close=previous,raw_close=close,raw_high=high,raw_low=low,
                       ask=ask,bid=bid,volume=volume,amount=amount,vwap=amount/volume if traded else np.nan,
                       status='reported_trade' if traded else 'no_reported_trade',source_lines=line+'\n'+lines[index+1])
        rows.append(row)
    result=pd.DataFrame(rows)
    if not result.empty and result.exchange_code.duplicated().any():raise ValueError('报价段落内证券代码重复')
    return result
