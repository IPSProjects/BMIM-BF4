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

    train_data = ips.configuration_selection.RandomSelection(
        data=filtered.atoms, n_configurations=1000
    )
    validation_data = ips.configuration_selection.RandomSelection(
        data=train_data.excluded_atoms, n_configurations=100
    )

    test_data = validation_data.excluded_atoms

    train_data = train_data.atoms

    model = ips.models.Apax(
        data=train_data,
        validation_data=validation_data.atoms,
        config="config/initial_model.yaml",
    )

    prediction = ips.analysis.Prediction(data=test_data, model=model)
    metrics = ips.analysis.PredictionMetrics(data=prediction)
    ips.analysis.EnergyHistogram(data=train_data, bins=100)
    ips.analysis.ForcesHistogram(data=train_data)

with project.group("ML0"):
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
        run_kwargs={"fmax": 0.01},
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


with project.group("ML1"):
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
        run_kwargs={"fmax": 0.01},
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

    ips.analysis.EnergyHistogram(data=train_data, bins=100)
    ips.analysis.ForcesHistogram(data=train_data)

    model = ips.models.Apax(
        data=train_data,
        validation_data=validation_data.atoms,
        config="config/initial_model.yaml",
    )

    prediction = ips.analysis.Prediction(data=test_data, model=model)
    metrics = ips.analysis.PredictionMetrics(data=prediction)


temperature_oszillator = ips.calculators.TemperatureOscillatingRampModifier(
    end_temperature=500, # decomp ~ 290
    start_temperature=230, # melting -75
    num_oscillations=10,
    temperature_amplitude=150,
)

eq_box_oszillator = ips.calculators.BoxOscillatingRampModifier(
    end_cell=14.58,
    cell_amplitude=1,
    num_oscillations=3,
)

with project.group("ML2"):
    md = ips.calculators.ASEMD(
        data=geo_opt.atoms,
        data_id=-1,
        model=model,
        thermostat=thermostat,
        modifier=[temperature_oszillator, eq_box_oszillator],
        checker_list=[uncertainty_check],
        steps=50000,
        sampling_rate=10,
    )

    geo_opt = ips.calculators.ASEGeoOpt(
        model=model,
        data=md.atoms,
        data_id=-1,
        optimizer="FIRE",
        run_kwargs={"fmax": 0.01},
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

    geo_opt = ips.calculators.ASEGeoOpt(
        model=cp2k,
        data=md.atoms,
        data_id=-1,
        optimizer="BFGS",
        run_kwargs={"fmax": 0.1},
    )

    

project.build()
