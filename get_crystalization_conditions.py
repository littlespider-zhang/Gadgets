import requests
import pandas as pd


from bs4 import BeautifulSoup

h = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36'}


def make_url(pdb, sub='experimental'):
    """
    Generate the url about experimental details on rcsb website
    corresponding to specific PDB id.
    """
    print(f"Url for {pdb.upper()}, generated...")
    return f"https://www.rcsb.org/{sub}/{pdb.upper()}"


def get_soup(pdb, sub='experimental'):
    """
    Construct BS object containing html doc on the given PDB id and sub.
    Return soup and pdb id
    """
    # Get html in text form
    url = make_url(pdb, sub)
    html_doc = requests.get(url, headers=h).text
    
    # Soup!
    soup = BeautifulSoup(html_doc, 'html.parser')
    
    #soup.prettify(), nicely show, but cannot use soup.title properly.
    print(f"HTML contents, got...")
    return soup


def find_table(in_soup, pdb, containing="Crystalization Experiments"):
    """
    Find the table(Tag object) containing crystalization conditions.
    Return table, pdb
    """
    # Find all tables in soup on given class_ ; Be careful with _ !
    tbs = in_soup.find_all('table',
                           # This is necessary
                           class_="table table-hover table-responsive")
    
    # Used for search
    find_condition = False
    
    # Search target table: 
    for tb in tbs:
        # check
        if not find_condition:
            
            # See if 'th's matches what we want
            all_headers = tb.find_all('th')
            for h in all_headers:
                if h.string == containing:
                    find_condition = True
                    break
        
        # Return the 1st table matches requirment.
        print(f"Table containing: {containing}, got...")
        return tb       


def collect_table_info(table, pdb, filename='crystalizaiton_conditions.txt'):
    """
    For output in tab sep file
    """
    # See website architecture
    output = []
    
    # Add headers
    headers = [i.string for i in table.contents[1].children]
    output.append(headers)
    
    # Add data, multiple rows possible
    for i in range(1, len(table.contents)-1):
        data = [i.string for i in table.contents[2].children]
        data[0] = pdb # Change id to pdb id
        output.append(data)
    
    print("Writing data to files...")
    print("Seperated by tab...")
    with open(filename, 'a', encoding='utf-8') as f:
        
        # Write by lines
        for line in output:
            for i in line:
                f.write(f"{i}\t")
            f.write('\n')
        print(f"Data written, see {filename}.\n\n")


def get_crystalization_conditions(pdb,f_name='crystalizaiton_conditions.txt'):
    """
    Get all info and write in a single file
    """
    # For requests
    h = {'User-Agent': 
         'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36'}
    
    # Get html content
    soup = get_soup(pdb, sub='experimental')
    
    # Find table containing crystalization conditions
    conditions_table = find_table(soup, pdb,
                                  containing="Crystalization Experiments")
    
    # Write files
    collect_table_info(conditions_table, pdb, filename=f_name)
    
    

if __name__ == "__main__":
    pdb_list = ['2RFK', '3HAY', '3LWR', '3LWQ','2HVY', '3HAX', '3LWP',
                '3LWO', '3LWV', '1SDS']
    
    for pdb in pdb_list:
        get_crystalization_conditions(pdb, f_name='crystalizaiton_conditions.txt')

    
        
    

            

