def auto_select_resi(chains, resi, name):
    """
    输入目标chain和残基编号，自动生成pymol的select命令
    输出为select name, (......)
    """
    pymol_selection = f'select {name}, '
    for c in chains:
        for r in resi:
            try:
                pymol_selection += f'(chain {c} and resi {r[0]}-{r[1]}) or '
            except:
                pymol_selection += f'(chain {c} and resi {r}) or '
    
    pymol_selection = pymol_selection[:-4]  # remove extra ' or'
    
    return pymol_selection

def auto_select_aa(chains, aa, name):

    """
    输入目标chain和残基类别，自动生成pymol的select命令
    输出为select name, (......)
    """
    pymol_selection = f'select {name}, '
    aa_types = ''
    for a in aa:
        aa_types += a + '+'
    for c in chains:
        pymol_selection += f'(chain {c} and resn {aa_types}) or '
    
    pymol_selection = pymol_selection[:-4]  # remove extra ' or'
    
    return pymol_selection

def auto_select_chains(objs, chains, name, default='select'):

    """
    输入目标chain和残基类别，自动生成pymol的select命令
    输出为select name, (......)
    """
    pymol_selection = f'{default} {name}, '
    for obj in objs:
        for c in chains:
            pymol_selection += f'({obj.strip()} and chain {c}) or '
    
    pymol_selection = pymol_selection[:-4]  # remove extra ' or'
    
    return pymol_selection



if __name__ == "__main__":
    name = 'cys'
    objs = ('Rvb1_2-6gej',)
    chains = ('U','W','Y',)
    resi = ((290,297),)
    resn = ('cys',)
    
    # objs = """yeast-chromatin_remodeller_6gej_cryoEM,yeast-chromatin_remodeller_6gen_cryoEM,yeast-INO80_hexasome_8ets_cryoEM,yeast-INO80_hexasome_8etu_cryoEM,yeast-INO80_hexasome_8etw_cryoEM,yeast-INO80_nucleosome_8eu9_cryoEM,yeast-INO80_nucleosome_8euf_cryoEM"""
    # objs = objs.split(',')


    command = auto_select_chains(objs, chains, name)
    print(command)
    
    
    print('\n\n')
    print(auto_select_resi(chains, resi, name))
    print('\n\n')
    print(auto_select_aa(chains, resn, name))
    print('\n\n')
    print(auto_select_chains(objs, chains, name))
    
    
