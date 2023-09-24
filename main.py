import ipsuite as ips

project = ips.Project(automatic_node_names=True)

cp2k = ips.calculators.CP2KSinglePoint(
    data=None,
    cp2k_params="config/cp2k.yaml",
    cp2k_files=["GTH_BASIS_SETS", "GTH_POTENTIALS", "dftd3.dat"],
)

with project.group("GeoOpt"):
    cation = ips.configuration_generation.SmilesToAtoms("CCCCN1C=C[N+](=C1)C")
    anion = ips.configuration_generation.SmilesToAtoms("[B-](F)(F)(F)F")

    single_structure = ips.configuration_generation.Packmol(
        data=[cation.atoms, anion.atoms],
        count=[1, 1],
        density=1210,  # 293 K
    )

    structure = ips.configuration_generation.Packmol(
        data=[single_structure.atoms],
        count=[10],
        density=1210,
    )

    geo_opt = ips.calculators.ASEGeoOpt(
        model=cp2k,
        data=structure.atoms,
        optimizer="BFGS",
        run_kwargs={"fmax": 0.1},
    )

with project.group("bootstrap"):
    rotate = ips.bootstrap.RotateMolecules(
        data=geo_opt.atoms,
        data_id=-1,
        n_configurations=15,
        maximum=10 * 3.1415 / 180,  # deg max rotation
        include_original=False,
        seed=1,
        model=cp2k,
    )
    translate = ips.bootstrap.TranslateMolecules(
        data=geo_opt.atoms,
        data_id=-1,
        n_configurations=15,
        maximum=0.3,  # Ang max molecular displacement
        include_original=False,
        seed=1,
        model=cp2k,
    )


project.build()
