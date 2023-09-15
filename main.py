import ipsuite as ips

project = ips.Project(automatic_node_names=True)
cp2k_shell = "mpirun -n 32 cp2k_shell.psmp"

with project.group("GeoOpt"):
    cation = ips.configuration_generation.SmilesToAtoms("CCCCN1C=C[N+](=C1)C")
    anion = ips.configuration_generation.SmilesToAtoms("[B-](F)(F)(F)F")

    single_structure = ips.configuration_generation.Packmol(
        data=[cation.atoms, anion.atoms],
        count=[1, 1],
        density=1210, # 293 K
    )

    structure = ips.configuration_generation.Packmol(
        data=[single_structure.atoms],
        count=[10],
        density=1210,
    )

    cp2k = ips.calculators.CP2KSinglePoint(
        data=structure.atoms,
        cp2k_params="config/cp2k.yaml",
        cp2k_files=["GTH_BASIS_SETS", "GTH_POTENTIALS", "dftd3.dat"],
        cp2k_shell=cp2k_shell,
    )

    geopt = ips.calculators.ASEGeoOpt(
        model=cp2k,
        data=structure.atoms,
        optimizer="BFGS",
        run_kwargs={"fmax": 0.1},
    )

with project.group("bootstrap"):
    rotate = ips.bootstrap.RotateMolecules(
        data=geopt.atoms,
        data_id=-1,
        n_configurations=15,
        maximum=5 * 3.1415 / 180,  # deg max rotation
        include_original=False,
        seed=1,
    )
    translate = ips.bootstrap.TranslateMolecules(
        data=geopt.atoms,
        data_id=-1,
        n_configurations=15,
        maximum=0.1,  # Ang max molecular displacement
        include_original=False,
        seed=1,
    )
    rattle = ips.bootstrap.RattleAtoms(
        data=geopt.atoms,
        data_id=-1,
        n_configurations=15,
        maximum=0.1,
        include_original=False,
        seed=1,
    )

    cp2k_rotate = ips.calculators.CP2KSinglePoint(
        data=rotate.atoms,
        cp2k_params="config/cp2k.yaml",
        cp2k_files=["GTH_BASIS_SETS", "GTH_POTENTIALS", "dftd3.dat"],
        cp2k_shell=cp2k_shell,
    )
    cp2k_translate = ips.calculators.CP2KSinglePoint(
        data=translate.atoms,
        cp2k_params="config/cp2k.yaml",
        cp2k_files=["GTH_BASIS_SETS", "GTH_POTENTIALS", "dftd3.dat"],
        cp2k_shell=cp2k_shell,
    )
    cp2k_rattle = ips.calculators.CP2KSinglePoint(
        data=rattle.atoms,
        cp2k_params="config/cp2k.yaml",
        cp2k_files=["GTH_BASIS_SETS", "GTH_POTENTIALS", "dftd3.dat"],
        cp2k_shell=cp2k_shell,
    )

project.build()
