"""Stock data source utilities."""

import re
from dataclasses import dataclass

# Predefined stock names by market
STOCK_NAMES = {
    "000001.SH": "上证指数",
    "399001.SZ": "深证成指",
    "000688.SH": "科创50",
    "399006.SZ": "创业板指",
    "159632.SZ": "纳斯达克ETF",
    "159870.SZ": "化工ETF",
    "159567.SZ": "港股创新药ETF",
    "600111.SH": "北方华创",
    "603986.SH": "兆易创新",
    "601138.SH": "工业富联",
    "002475.SZ": "立讯精密",
    "002156.SZ": "通富微电",
    "000021.SZ": "深科技",
    "601020.SH": "华钰矿业",
    "600036.SH": "招商银行",
    "000333.SZ": "美的集团",
    "603191.SH": "望变电气",
    "600089.SH": "特变电工",
    "601288.SH": "农业银行",
    "600887.SH": "伊利股份",
    "600900.SH": "长江电力",
    "600362.SH": "江西铜业",
    "000807.SZ": "云铝股份",
    "000792.SZ": "盐湖股份",
    "HSTECH.HK": "恒生科技指数",
    "9988.HK": "阿里巴巴",
    "1024.HK": "快手",
    "0981.HK": "中芯国际",
    "9961.HK": "携程集团",
    "3690.HK": "美团",
    "1810.HK": "小米集团",
    "3750.HK": "携程-S",
    "9880.HK": "小鹏汽车",
    "0700.HK": "腾讯控股",
    "2097.HK": "蜜雪集团",
    "9868.HK": "零跑汽车",
    "1357.HK": "美图",
    "0100.HK": "MiniMax",
    "6082.HK": "壁仞科技",
    "2577.HK": "英诺赛科",
    "2020.HK": "安踏体育",
    "0522.HK": "ASM太平洋",
    "1347.HK": "华虹半导体",
    "9626.HK": "贝壳-W",
    "AMZN": "亚马逊",
    "MSFT": "微软",
    "TSLA": "特斯拉",
    "AAPL": "苹果",
    "ASML": "阿斯麦",
    "TSM": "台积电",
    "SE": "Sea Ltd",
    "SMR": "NuScale Power",
    "CRDO": "Credo Technology",
    "OKLO": "Oklo Inc",
    "QCOM": "高通",
    "AMD": "超威半导体",
    "INTC": "英特尔",
    "GOOGL": "谷歌",
    "AVGO": "博通",
    "NVDA": "英伟达",
    "PONY": "小马智行",
    "PDD": "拼多多",
    "CRWV": "Coreweave",
    "MU": "美光科技",
    "SNDK": "西部数据",
    "UNH": "联合健康",
    "TCEHY": "腾讯ADR",
    "NBIS": "Nebius Group",
}

ZONE_SUFFIX = {
    "cn": [".SH", ".SZ"],
    "hk": [".HK"],
    "us": [],
}


@dataclass
class StockInfo:
    """Stock information container."""
    code: str
    market: str  # 'a_share', 'hk_share', 'us_share'
    exchange: str  # 'SZ', 'SH', 'HK', 'NASDAQ', 'NYSE'
    symbol: str  # Original symbol without exchange


class StockInfoResolver:
    """Resolve stock code to market and exchange."""

    @staticmethod
    def resolve(stock_code: str) -> StockInfo:
        stock_code = stock_code.upper().strip()

        a_share_match = re.match(r'^(\d{6})\.(SZ|SH)$', stock_code)
        if a_share_match:
            return StockInfo(
                code=stock_code,
                market='a_share',
                exchange=a_share_match.group(2),
                symbol=a_share_match.group(1)
            )

        hk_match = re.match(r'^(\d{4,5})\.HK$', stock_code)
        if hk_match:
            return StockInfo(
                code=stock_code,
                market='hk_share',
                exchange='HK',
                symbol=hk_match.group(1)
            )

        if '.' not in stock_code:
            if re.match(r'^[A-Z]{1,5}$', stock_code):
                return StockInfo(
                    code=stock_code,
                    market='us_share',
                    exchange='NASDAQ',
                    symbol=stock_code
                )

        raise ValueError(f"Unknown stock code format: {stock_code}")

    @staticmethod
    def get_index_code(stock_info: StockInfo) -> str:
        if stock_info.market == 'a_share':
            return '000001.SH'
        elif stock_info.market == 'hk_share':
            return '^HSI'
        elif stock_info.market == 'us_share':
            return '^GSPC'
        return '^GSPC'

    @staticmethod
    def get_name(stock_code: str) -> str:
        return STOCK_NAMES.get(stock_code.upper(), stock_code)


def format_stock_code(code: str, exchange: str) -> str:
    if exchange in ['SZ', 'SH']:
        return f"{code}.{exchange}"
    elif exchange == 'HK':
        return f"{code}.HK"
    return code
