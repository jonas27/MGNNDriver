import json
from typing import List

import requests
from requests.structures import CaseInsensitiveDict

from graphdriver.utils import paths


def neighbors(cancer: str, genes: List):
    paths


def single(gene: str, genes: list):
    url = "https://www.genefriends.org/api/validate"

    headers = CaseInsensitiveDict()
    headers["Accept"] = "application/json, text/plain, */*"
    headers["Accept-Language"] = "en-US,en;q=0.9"
    headers["Connection"] = "keep-alive"
    headers["Content-Type"] = "application/json"
    headers["Origin"] = "https://www.genefriends.org"
    headers["Referer"] = "https://www.genefriends.org/start/input"
    headers["Sec-Fetch-Dest"] = "empty"
    headers["Sec-Fetch-Mode"] = "cors"
    headers["Sec-Fetch-Site"] = "same-origin"
    headers[
        "User-Agent"
    ] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36"
    headers["sec-ch-ua-mobile"] = "?0"
    data = '{"seedGenes":[{}],"species":"ENS","dataSource":"SRA","tissue":"G0","objectType":"G"}'.format(gene)

    resp_pre = requests.post(url, headers=headers, data=data)
    resp_data = resp_pre.json()[0]
    req_data = {"seedGenesInDatabaseObj": resp_data, "species": "ENS", "dataSource": "SRA", "tissue": "G0", "threshold": 0.5}

    url = "https://www.genefriends.org/api/find-friends-metadata-single-gene"
    headers["Referer"] = "https://www.genefriends.org/start/setup"
    resp = requests.post(url, headers=headers, data=json.dumps(req_data))
    neighs = [(neighbor["symbol"], neighbor["correlation"]) for neighbor in resp.json()[:15]]
    return neighs
