import pandas as pd 

fdi = pd.read_csv("fdi_subsectores.csv")
paises = pd.read_csv("paises_iso_code.csv")


nombres_paises_no_coinciden = set(fdi["Source country"]) - set(paises["category_name"]).intersection(fdi["Source country"])

fdi["iso_code3"] = fdi["Source country"].replace({i:j for i,j in paises[["category_name", "iso_alpha_3"]].to_records(index = False)})

iso_faltan = {
 'Bosnia-Herzegovina' : 'BIH',
 'Brunei' : 'BRN',
 'Cote d Ivoire' : 'CIV',
 'Czech Republic' : 'CZE',
 'Democratic Republic of Congo' : 'COD',
 'Hong Kong' : 'HKG',
 'Macau' : 'MAC',
 'Moldova' : 'MDA',
 'Palestine' : 'PSE',
 'Republic of Kosovo' : 'UNK',
 'Russia' : 'RUS',
 'Somaliland' : 'SOM',
 'South Korea' : 'KOR',
 'Tanzania' : 'TZA',
 'Trinidad & Tobago' : 'TTO',
 'UAE' : 'ARE',
 'United States' : 'USA',
 'Vietnam' : 'VNM'
 }

fdi["iso_code3"] = fdi["iso_code3"].replace(iso_faltan)