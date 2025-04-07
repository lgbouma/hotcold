# n_H hydrogen
from astropy import units as u, constants as c

lambda_halpha = 6563*u.Angstrom
nu_halpha = c.c / lambda_halpha
hnu_Halpha = c.h * nu_halpha
n_H_pref = (
    2
    * (1e27 * u.erg/u.s)**(1/2)
    * (1e-13 * u.cm**3/u.s)**(-1/2)
    * (hnu_Halpha)**(-1/2)
    * (3 / (4*3.141) )**(1/2)
    * (1 / (0.1*u.Rsun) )**(3/2)
)
print(f"n_H = {n_H_pref.cgs:.2e}")

R = 0.1*u.Rsun
V = 4*3.14/3 * R**3
M_gas = n_H_pref * c.m_p * V

print(f"M_gas = {M_gas.cgs:.2e}")

B_pref = (
    (8*3.141*c.k_B.cgs)**(1/2)
    * (n_H_pref.cgs * 3000*u.K)**(1/2)
)
magnetic_field_equiv = [(u.erg**0.5/u.cm**1.5, u.G)]
print(f"B = {B_pref.to(u.G, equivalencies=magnetic_field_equiv):.2e}")

