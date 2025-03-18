from astropy import units as u, constants as c

x = 0.5
L_Hα_star = 1e28 * u.erg / u.s # from the STAR (!)
L_Hα_clump = (1/5) * 1e28 * u.erg / u.s # from the STAR (!)

λ = 6365 * u.Angstrom
ν = c.c / λ

E = c.h * ν

Rstar = 0.4 * u.Rsun
r = 0.1 * Rstar
V = 4 * 3.14/3 * r**3

alpha_eff_Halpha = 1e-13 * u.cm**3 / u.s

n_H = (
    (1/x) *
    (L_Hα_clump /
     (alpha_eff_Halpha * E * V)
    )**(0.5)
)

print(f"n_H: {n_H.cgs:.2e}")

M_H = n_H * c.m_p * V
print(f"M_H: {M_H.cgs:.2e}")
