import ipsuite as ips

project = ips.Project(automatic_node_names=True)


thermostat = ips.calculators.LangevinThermostat(
    temperature=298.15, friction=0.01, time_step=0.5
)

uncertainty_check = ips.analysis.ThresholdCheck(
    value="forces_uncertainty", max_value=0.5, larger_only=True
)

with project.group("classical"):
    data = ips.AddData("data/nvt_eq.xyz")
    cp2k = ips.calculators.CP2KSinglePoint(
        data=data.atoms,
        cp2k_params="config/cp2k.yaml",
        cp2k_files=["GTH_BASIS_SETS", "GTH_POTENTIALS", "dftd3.dat"],
    )

    filtered = ips.configuration_selection.FilterOutlier(data=cp2k.atoms)

    test_data = ips.configuration_selection.RandomSelection(
        data=filtered.atoms, n_configurations=500
    )

    validation_data = ips.configuration_selection.RandomSelection(
        data=test_data.excluded_atoms, n_configurations=100
    )

    train_data = ips.configuration_selection.RandomSelection(
        data=validation_data.excluded_atoms, n_configurations=100
    )

    model = ips.models.Apax(
        data=train_data.atoms,
        validation_data=validation_data.atoms,
        config="config/initial_model.yaml",
    )

    prediction = ips.analysis.Prediction(data=test_data, model=model)
    metrics = ips.analysis.PredictionMetrics(data=prediction)
    ips.analysis.EnergyHistogram(data=train_data, bins=100)
    ips.analysis.ForcesHistogram(data=train_data)

    md = ips.calculators.ASEMD(
        data=data.atoms,
        data_id=-1,
        model=model,
        thermostat=thermostat,
        checker_list=[uncertainty_check],
        steps=50000,
        sampling_rate=10,
    )

with project.group("ML0"):
    kernel_selection = ips.models.apax.BatchKernelSelection(
        data=train_data.excluded_atoms,
        train_data=train_data.atoms,
        models=model,
        n_configurations=400,
        processing_batch_size=4,
    )

    train_data = train_data.atoms + kernel_selection.atoms

    model = ips.models.Apax(
        data=train_data,
        validation_data=validation_data.atoms,
        config="config/initial_model.yaml",
    )

    prediction = ips.analysis.Prediction(data=test_data, model=model)
    metrics = ips.analysis.PredictionMetrics(data=prediction)
    ips.analysis.EnergyHistogram(data=train_data, bins=100)
    ips.analysis.ForcesHistogram(data=train_data)

with project.group("ML1"):
    md = ips.calculators.ASEMD(
        data=data.atoms,
        data_id=-1,
        model=model,
        thermostat=thermostat,
        checker_list=[uncertainty_check],
        steps=50000,
        sampling_rate=10,
    )

    geo_opt = ips.calculators.ASEGeoOpt(
        model=model,
        data=md.atoms,
        data_id=-1,
        optimizer="FIRE",
        run_kwargs={"fmax": 0.5},
    )

    kernel_selection = ips.models.apax.BatchKernelSelection(
        data=md.atoms + geo_opt.atoms,
        train_data=train_data,
        models=model,
        n_configurations=50,
        processing_batch_size=4,
    )

    cp2k = ips.calculators.CP2KSinglePoint(
        data=kernel_selection.atoms,
        cp2k_params="config/cp2k.yaml",
        cp2k_files=["GTH_BASIS_SETS", "GTH_POTENTIALS", "dftd3.dat"],
    )

    train_data += cp2k.atoms

    model = ips.models.Apax(
        data=train_data,
        validation_data=validation_data.atoms,
        config="config/initial_model.yaml",
    )

    prediction = ips.analysis.Prediction(data=test_data, model=model)
    metrics = ips.analysis.PredictionMetrics(data=prediction)
    ips.analysis.EnergyHistogram(data=train_data, bins=100)
    ips.analysis.ForcesHistogram(data=train_data)

for idx in range(2, 5):
    with project.group(f"ML{idx}"):
        md = ips.calculators.ASEMD(
            data=geo_opt.atoms,
            data_id=-1,
            model=model,
            thermostat=thermostat,
            checker_list=[uncertainty_check],
            steps=50000,
            sampling_rate=10,
        )

        geo_opt = ips.calculators.ASEGeoOpt(
            model=model,
            data=md.atoms,
            data_id=-1,
            optimizer="FIRE",
            run_kwargs={"fmax": 0.5},
        )

        kernel_selection = ips.models.apax.BatchKernelSelection(
            data=md.atoms + geo_opt.atoms,
            train_data=train_data,
            models=model,
            n_configurations=50,
            processing_batch_size=4,
        )

        cp2k = ips.calculators.CP2KSinglePoint(
            data=kernel_selection.atoms,
            cp2k_params="config/cp2k.yaml",
            cp2k_files=["GTH_BASIS_SETS", "GTH_POTENTIALS", "dftd3.dat"],
        )

        train_data += cp2k.atoms
        # if idx > 3:
        #     break

        model = ips.models.Apax(
            data=train_data,
            validation_data=validation_data.atoms,
            config="config/initial_model.yaml",
        )

        prediction = ips.analysis.Prediction(data=test_data, model=model)
        metrics = ips.analysis.PredictionMetrics(data=prediction)
        ips.analysis.EnergyHistogram(data=train_data, bins=100)
        ips.analysis.ForcesHistogram(data=train_data)

temperature_oszillator = ips.calculators.TemperatureOscillatingRampModifier(
    end_temperature=500,  # decomp ~ 290
    start_temperature=230,  # melting -75
    num_oscillations=10,
    temperature_amplitude=150,
)


for idx in range(5, 8):
    with project.group(f"ML{idx}") as grp:
        md = ips.calculators.ASEMD(
            data=geo_opt.atoms,
            data_id=-1,
            model=model,
            thermostat=thermostat,
            modifier=[temperature_oszillator],
            checker_list=[uncertainty_check],
            steps=50000,
            sampling_rate=10,
        )
        geo_opt = ips.calculators.ASEGeoOpt(
            model=model,
            data=md.atoms,
            data_id=-1,
            optimizer="FIRE",
            run_kwargs={"fmax": 0.5},
        )

        kernel_selection = ips.models.apax.BatchKernelSelection(
            data=md.atoms + geo_opt.atoms,
            train_data=train_data,
            models=model,
            n_configurations=50,
            processing_batch_size=4,
        )


        cp2k = ips.calculators.CP2KSinglePoint(
            data=kernel_selection.atoms,
            cp2k_params="config/cp2k.yaml",
            cp2k_files=["GTH_BASIS_SETS", "GTH_POTENTIALS", "dftd3.dat"],
        )

        train_data += cp2k.atoms

        model = ips.models.Apax(
            data=train_data,
            validation_data=validation_data.atoms,
            config="config/initial_model.yaml",
        )

        prediction = ips.analysis.Prediction(data=test_data, model=model)
        metrics = ips.analysis.PredictionMetrics(data=prediction)
        ips.analysis.EnergyHistogram(data=train_data, bins=100)
        ips.analysis.ForcesHistogram(data=train_data)



eq_box_oszillator = ips.calculators.BoxOscillatingRampModifier(
    end_cell=14.58,
    cell_amplitude=1,
    num_oscillations=3,
)

for idx in range(8, 10):
    with project.group(f"ML{idx}") as grp:
        md = ips.calculators.ASEMD(
            data=geo_opt.atoms,
            data_id=-1,
            model=model,
            thermostat=thermostat,
            modifier=[eq_box_oszillator],
            checker_list=[uncertainty_check],
            steps=50000,
            sampling_rate=10,
        )
        geo_opt = ips.calculators.ASEGeoOpt(
            model=model,
            data=md.atoms,
            data_id=-1,
            optimizer="FIRE",
            run_kwargs={"fmax": 0.5},
        )

        kernel_selection = ips.models.apax.BatchKernelSelection(
            data=md.atoms + geo_opt.atoms,
            train_data=train_data,
            models=model,
            n_configurations=50,
            processing_batch_size=4,
        )

        cp2k = ips.calculators.CP2KSinglePoint(
            data=kernel_selection.atoms,
            cp2k_params="config/cp2k.yaml",
            cp2k_files=["GTH_BASIS_SETS", "GTH_POTENTIALS", "dftd3.dat"],
        )

        train_data += cp2k.atoms

        if idx > 8:
            break

        model = ips.models.Apax(
            data=train_data,
            validation_data=validation_data.atoms,
            config="config/initial_model.yaml",
        )

        prediction = ips.analysis.Prediction(data=test_data, model=model)
        metrics = ips.analysis.PredictionMetrics(data=prediction)
        ips.analysis.EnergyHistogram(data=train_data, bins=100)
        ips.analysis.ForcesHistogram(data=train_data)

    eq_box_oszillator.num_ramp_oscillations = 0.5

project.build()
