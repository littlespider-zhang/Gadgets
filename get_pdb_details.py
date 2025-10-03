import requests
import re

h = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36'}


def make_url(pdb, sub='Annotations'):
    return f"https://www.rcsb.org/{sub}/{pdb.upper()}"


def get_text(pdb, sub='Annotations'):
    url = make_url(pdb, sub)
    html = requests.get(url, headers=h)
    return html.text


def get_title(pdb):
    html_text = get_text(pdb)
    p = '(?<=<h4><span>).+?(?=</span></h4>)'
    title = re.findall(p, html_text)

    return title[0]


def get_experimental_method(pdb):
    html_text = get_text(pdb, sub='experimental')
    m_pattern = '(?<="experimentalMethod">).+?(?=<hr></div>)'
    m = re.findall(m_pattern, html_text)[0]
    print(f"ID: {pdb.upper()}, Method: {m}")
    return m, html_text


def get_xray_details(pdb):
    method, text = get_experimental_method(pdb)
    if method == 'X-RAY DIFFRACTION':
        p = '(?<=Crystalization Experiments).+?(?=Crystal Data)'
        content = re.findall(p, text)
        p_2 = '(?<=<t[hd]>).*?(?=</t[hd]>)'
        details = re.findall(p_2, content[0])

        return details
    else:
        return None

# pdb_list = ['2RFK', '3HAY', '3LWR', '3LWQ', '2HVY', '3HAX', '3LWP', '3LWO', '3LWV', '1SDS']
# crystal_condition = {}
# for id in pdb_list:
#     url = experiment_url(id)
#     print(url)
#     html_text = get_text(url)
#     crystal_condition[id] = {'title': get_title(html_text),
#                              'details': get_experiment_details(html_text)}
