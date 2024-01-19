import ipsuite as ips 
import zntrack

project = ips.Project(automatic_node_names=True)

ramp_density = ips.calculators.RescaleBoxModifier(
    density=1162
)
thermostat = ips.calculators.LangevinThermostat(
    temperature=353, friction=0.01, time_step=0.5
)

mlp = zntrack.from_rev("ML17_training_MLModel_2", rev="ml17", remote="https://github.com/IPSProjects/BMIM-BF4_new.git")

with project.group("depl") as depl:
    anion = ips.configuration_generation.SmilesToAtoms(
        smiles="[B-](F)(F)(F)F"
    )
    cation = ips.configuration_generation.SmilesToAtoms(
        smiles="CCCCN1C=C[N+](=C1)C"
    )

    single_structure = ips.configuration_generation.Packmol(
        data=[cation.atoms, anion.atoms],
        count=[1, 1],
        density=1210,
        pbc=False,
    )

    structure = ips.configuration_generation.Packmol(
        data=[single_structure.atoms],
        count=[16], # 16, 32, 64
        density=900,
    )

    geo_opt = ips.calculators.ASEGeoOpt(
        model=mlp,
        data=structure.atoms,
        data_id=-1,
        optimizer="FIRE",
        run_kwargs={"fmax": 0.05},
    )

    density_md = ips.calculators.ASEMD(
        data=geo_opt.atoms,
        data_id=-1,
        model=mlp,
        modifier=[ramp_density],
        thermostat=thermostat,
        steps=10_000,
        sampling_rate=10,
    )

    # relaxation 

    md = ips.calculators.ASEMD(
        data=density_md.atoms,
        data_id=-1,
        model=mlp,
        thermostat=thermostat,
        steps=10_000,
        sampling_rate=10,
    )

project.build(nodes=[depl])
