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


with project.group("initial_model"):
    data = cp2k_rotate.atoms + cp2k_translate.atoms + cp2k_rattle.atoms
    train_data_bootstrap = ips.configuration_selection.RandomSelection(data=data, n_configurations=30)
    geoopt_data = ips.configuration_selection.IndexSelection(data = geopt.atoms, indices=slice(10, None, None))

    train_data_geoopt = ips.configuration_selection.RandomSelection(data=geoopt_data, n_configurations=50)
    train_data = train_data_bootstrap.atoms + train_data_geoopt.atoms
    validation_data = train_data_bootstrap.excluded_atoms + train_data_geoopt.excluded_atoms
    

    seed_model_1 = ips.models.Apax(
            data=train_data,
            validation_data=validation_data,
            config="config/initial_model_1.yaml",
        )
    seed_model_2 = ips.models.Apax(
            data=train_data,
            validation_data=validation_data,
            config="config/initial_model_2.yaml",
        )

    seed_model = ips.models.EnsembleModel(models=[seed_model_1, seed_model_2])
    
    prediction = ips.analysis.Prediction(data=data, model=seed_model)
    metrics = ips.analysis.PredictionMetrics(data=prediction)

# with project.group("AL0") as al0:


project.build()
