import ipsuite as ips
import zntrack

project = ips.Project(automatic_node_names=True)

model = zntrack.from_rev(
    "SPICE_pro_neut_E0", remote="/ssd/fzills/IPS/MACE_DFT/MACE_neutral_cleaned_DFT_E0"
)

thermostat = ips.calculators.LangevinThermostat(
    temperature=298.15, friction=0.01, time_step=0.5
)

with project.group("classical"):
    cation = ips.configuration_generation.SmilesToAtoms("CCCCN1C=C[N+](=C1)C")
    anion = ips.configuration_generation.SmilesToAtoms("C(=[N-])=NC#N")

    single_structure = ips.configuration_generation.Packmol(
        data=[cation.atoms, anion.atoms],
        count=[1, 1],
        density=1060,
    )

    structure = ips.configuration_generation.Packmol(
        data=[single_structure.atoms],
        count=[10],
        density=1060,
    )
    geopt = ips.calculators.ASEGeoOpt(
        model=model,
        data=structure.atoms,
        optimizer="BFGS",
        run_kwargs={"fmax": 0.1},
    )

    md = ips.calculators.ASEMD(
        data=geopt.atoms,
        data_id=-1,
        model=model,
        thermostat=thermostat,
        steps=100,
        sampling_rate=1,
    )

project.build()
