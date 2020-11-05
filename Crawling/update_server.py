from requests import post

if __name__ == "__main__":
    headers = {"Content-Type": "application/json"}
    address = "http://127.0.0.1:2431/update"
    company = post(address, headers=headers)
    print(str(company.content, encoding="utf-8"))
