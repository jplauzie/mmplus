import numpy as np

from mumaxplus import Ferromagnet, Grid, World

c = 4e-9
A = 10e-12
Ms = 800e3

def run(m3, openbc):
    world = World(cellsize=(c, c, c))
    magnet = Ferromagnet(world, Grid((4, 1, 1)))
    magnet.enable_demag = False
    magnet.enable_openbc = openbc
    magnet.aex = A
    magnet.msat.set(lambda x, y, z: Ms if x < 2 * c else 0.0)  # check the printout below

    m = np.zeros((3, 1, 1, 4))
    m[:, 0, 0, 0] = (1, 0, 0)
    m[:, 0, 0, 1] = (0, 1, 0)
    m[:, 0, 0, 2] = (0, 0, 1)
    m[:, 0, 0, 3] = m3          # stored m of the EMPTY cell (Ms = 0)
    magnet.magnetization = m

    print(f"openbc={openbc}, m(cell 3)={m3}")
    print("  Ms:", magnet.msat.eval()[0, 0, 0, :])
    B = magnet.exchange_field.eval()
    for i in range(4):
        print(f"  B_ex cell {i}: {B[:, 0, 0, i]}")
    print("  exchange_energy:", magnet.exchange_energy.eval())

for openbc in (False, True):
    run((0, 0, 1), openbc)
    run((0, 0, -1), openbc)