import pandas as pd
import requests
import os

vaccine_url = 'https://data.ontario.ca/dataset/752ce2b7-c15a-4965-a3dc-397bf405e7cc/resource/775ca815-5028-4e9b-9dd4' \
      '-6975ff1be021/download/vaccines_by_age.csv'
vaccine_request = requests.get(vaccine_url)
path = os.getcwd()[:-5] + 'Model Dependencies/'

open(path + 'vaccine_by_age_auto_update.csv', "wb").write(vaccine_request.content)


admin_vaccine_url = 'https://data.ontario.ca/dataset/752ce2b7-c15a-4965-a3dc-397bf405e7cc/resource/8a89caa9-511c-' \
                    '4568-af89-7f2174b4378c/download/vaccine_doses.csv'
admin_vaccine_request = requests.get(admin_vaccine_url)
path = os.getcwd()[:-5] + 'Model Dependencies/'

open(path + 'vaccine_by_age_admin.csv', "wb").write(admin_vaccine_request.content)