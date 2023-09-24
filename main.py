import ipsuite as ips

project = ips.Project(automatic_node_names=True)

cp2k = ips.calculators.CP2KSinglePoint(
    data=None,
    cp2k_params="config/cp2k.yaml",
    cp2k_files=["GTH_BASIS_SETS", "GTH_POTENTIALS", "dftd3.dat"],
)

mapping = ips.geometry.BarycenterMapping(data=None)

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

    volume_scan = ips.analysis.BoxScale(
        data=geo_opt.atoms,
        mapping=mapping,
        model=cp2k,
        start=0.9,
        stop=2.0,
        num=30,
        data_id=-1,
    )

with project.group("initial_model") as initial_model:
    data = rotate.atoms + translate.atoms + volume_scan.atoms  # should be 60 in total
    train_data_bootstrap = ips.configuration_selection.RandomSelection(
        data=data, n_configurations=30
    )
    geoopt_data = ips.configuration_selection.IndexSelection(
        data=geo_opt.atoms, indices=slice(10, None, None)
    )

    train_data_geoopt = ips.configuration_selection.RandomSelection(
        data=geoopt_data, n_configurations=100
    )
    train_data = train_data_bootstrap.atoms + train_data_geoopt.atoms
    validation_data = (
        train_data_bootstrap.excluded_atoms + train_data_geoopt.excluded_atoms
    )

    model_1 = ips.models.Apax(
        data=train_data,
        validation_data=validation_data,
        config="config/initial_model_1.yaml",
    )
    model_2 = ips.models.Apax(
        data=train_data,
        validation_data=validation_data,
        config="config/initial_model_2.yaml",
    )

    model = ips.models.ApaxEnsemble(models=[model_1, model_2])

    prediction = ips.analysis.Prediction(data=validation_data, model=model)
    metrics = ips.analysis.PredictionMetrics(data=prediction)


thermostat = ips.calculators.LangevinThermostat(
    temperature=293, friction=0.01, time_step=0.5
)

uncertainty_check = ips.analysis.ThresholdCheck(
    value="energy_uncertainty", max_value=2.0, larger_only=True
)

models = []

for loop in range(5):
    with project.group(f"AL{loop}", "train") as altrain:
        md = ips.calculators.ASEMD(
            data=geo_opt.atoms if loop == 0 else ref_geopt_train.atoms,
            data_id=-1,
            model=model,
            thermostat=thermostat,
            checker_list=[uncertainty_check],
            steps=100000,
            sampling_rate=1,
        )

        model_geopt = ips.calculators.ASEGeoOpt(
            model=model,
            data=md.atoms,
            data_id=-5,
            optimizer="BFGS",
            run_kwargs={"fmax": 0.5},
            checker_list=[uncertainty_check],
        )

        ref_geopt_train = ips.calculators.ASEGeoOpt(
            model=cp2k,
            data=model_geopt.atoms,
            data_id=-1,
            optimizer="BFGS",
            run_kwargs={"fmax": 0.5},
        )

        md_selection = ips.configuration_selection.IndexSelection(
            data=md.atoms, indices=slice(0, -1)
        )
        confs = ips.configuration_selection.RandomSelection(
            data=md_selection.atoms, n_configurations=15
        )

        cp2k_train = ips.calculators.CP2KSinglePoint(
            data=confs.atoms,
            cp2k_params="config/cp2k.yaml",
            cp2k_files=["GTH_BASIS_SETS", "GTH_POTENTIALS", "dftd3.dat"],
        )

        train_data += cp2k_train.atoms
        train_data += ref_geopt_train.atoms

    with project.group(f"AL{loop}", "test") as altest:
        md = ips.calculators.ASEMD(
            data=geo_opt.atoms if loop == 0 else ref_geopt_test.atoms,
            data_id=-10 if loop == 0 else -1,  # slightly different start configuration
            model=model,
            thermostat=thermostat,
            checker_list=[uncertainty_check],
            steps=100000,
            sampling_rate=1,
        )

        model_geopt = ips.calculators.ASEGeoOpt(
            model=model,
            data=md.atoms,
            data_id=-5,
            optimizer="BFGS",
            run_kwargs={"fmax": 0.5},
            checker_list=[uncertainty_check],
        )

        ref_geopt_test = ips.calculators.ASEGeoOpt(
            model=cp2k,  # better restart wavefunction
            data=model_geopt.atoms,
            data_id=-1,
            optimizer="BFGS",
            run_kwargs={"fmax": 0.5},
        )

        md_selection = ips.configuration_selection.IndexSelection(
            data=md.atoms, indices=slice(0, -1)
        )
        confs = ips.configuration_selection.RandomSelection(
            data=md_selection.atoms, n_configurations=5
        )

        cp2k_test = ips.calculators.CP2KSinglePoint(
            data=confs.atoms,
            cp2k_params="config/cp2k.yaml",
            cp2k_files=["GTH_BASIS_SETS", "GTH_POTENTIALS", "dftd3.dat"],
        )

        validation_data += ref_geopt_test.atoms
        validation_data += cp2k_test.atoms

    with project.group(f"AL{loop}", "model") as almodel:
        model_1 = ips.models.Apax(
            data=train_data,
            validation_data=validation_data,
            config="config/initial_model_1.yaml",
        )
        model_2 = ips.models.Apax(
            data=train_data,
            validation_data=validation_data,
            config="config/initial_model_2.yaml",
        )

        model = ips.models.ApaxEnsemble(models=[model_1, model_2])
        models.append(model)

for loop in range(5):
    with project.group(f"AL{loop}", "metrics"):
        prediction = ips.analysis.Prediction(data=validation_data, model=models[loop])
        metrics = ips.analysis.PredictionMetrics(data=prediction)


project.build()
