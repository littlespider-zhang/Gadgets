import math
# parameters
c = 3 * 10**8 # m/sec light speed
m0 = 9.11 * 10**(-31) # kg 电子静止质量
e = 1.6 * 10**(-19) # C 电荷量的绝对值
h = 6.62 * 10**(-34) # J*s 普朗克常数

def velocity(V=0):
    # V 加速电压
    global c,m0,e
    V = V*1000
    a = m0*c*c/(e*V + m0*c*c)
    b = 1 - a**2
    v = c * math.sqrt(b)
    print(f"velocity of electron with voltage {V} is {v} m/sec")
    print(f"v/c -> {round(v/c,3)}")
    return v

def wavelength(V=0):
    # V 加速电压
    global c,m0,e,h
    
    v = velocity(V)
    a = h**2 * (c**2 - v**2)
    b = m0**2 * v**2 * c**2
    w = math.sqrt(a/b)
    print(f"wavelength of electron with voltage {V} is {w*10**9} nm")
    return w*10**9

def point_resolution(V=0,cs=2.7):
    """cs in mm, wavelength in nm"""
    w = wavelength(V)
    # convert unit to nanometer
    rd = 0.65 * (cs*10**6)**(0.25) * (w*10**9)**(0.75) # nm
    print(f"point resolution is {round(rd*10,4)} angstrom.")
    return rd

def scherzer_defocus(cs=2.7, V=0):
    """cs in mm, wavelength in nm"""
    w = wavelength(V)
    del_f = -1.2 * (cs*10**6*w)**0.5 # nm
    print(f"scherzer_defocus is {del_f*10} angstrom \nwith Cs = {cs} mm")
    return del_f

def information_limit(V=300,half_weight=0.4, cc=2.7):
    # V 加速电压 kV
    # half_weight 半高宽 0.4eV
    global c,m0,e
    V = V*1000
    
    a = m0*c*c/(half_weight*e*V + m0*c*c) # different!!!!!
    b = 1 - a**2
    v = c * math.sqrt(b)
    print(v)
    a = h**2 * (c**2 - v**2)
    b = m0**2 * v**2 * c**2
    w = math.sqrt(a/b)
    print(w)
    D = cc*10**(-3)*half_weight/V
    print(D)
    delta_cc = math.sqrt(math.pi*w*D/2)
    print(f"Information limit of {V/1000} kV, {half_weight} eV, Cc={cc}:")
    print(f"\t>{delta_cc*10**9} nm")
    


#scherzer_defocus(2.7,300)
#information_limit(V=120, half_weight=0.4)
scherzer_defocus(cs=2.7, V=300)
    
