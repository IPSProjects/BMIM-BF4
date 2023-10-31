from dscribe.descriptors import SOAP
import zntrack
import numpy as np

geoopt = zntrack.from_rev("GeoOpt_ASEGeoOpt")
# print(geoopt.atoms)


soap = SOAP(
    species=list(set(geoopt.atoms[0].get_atomic_numbers())),
    periodic=True,
    r_cut=6.0,
    n_max=7,
    l_max=5,
    sigma=0.5,
)

soap = soap.create(geoopt.atoms)
print(soap)
np.save("soap.npy", soap)