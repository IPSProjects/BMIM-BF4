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

    filtered = ips.configuration_selection.FilterOutlier(
        data=cp2k.atoms
    )

    train_data = ips.configuration_selection.RandomSelection(
        data=filtered.atoms, n_configurations=1000
    )
    validation_data = ips.configuration_selection.RandomSelection(
        data=train_data.excluded_atoms, n_configurations=100
    )

    test_data = validation_data.excluded_atoms

    model = ips.models.Apax(
        data=train_data.atoms,
        validation_data=validation_data.atoms,
        config="config/initial_model.yaml",
    )

    prediction = ips.analysis.Prediction(
        data=test_data, model=model
    )
    metrics = ips.analysis.PredictionMetrics(
        data=prediction
    )

    # train_data_selection = ips.configuration_selection.RandomSelection(
    #     cp2k.atoms, n_configurations=1000
    # )
    # train_data = train_data_selection.atoms
    # test_data = train_data_selection.excluded_atoms



project.build()

