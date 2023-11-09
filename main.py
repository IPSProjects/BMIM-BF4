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

    train_data_selection = ips.configuration_selection.RandomSelection(
        cp2k.atoms, n_configurations=1000
    )
    train_data = train_data_selection.atoms
    test_data = train_data_selection.excluded_atoms

with project.group("ML0"):
    model1 = ips.models.Apax(
        data=train_data,
        validation_data=test_data,
        config="config/initial_model_1.yaml",
    )
    model2 = ips.models.Apax(
        data=train_data,
        validation_data=test_data,
        config="config/initial_model_2.yaml",
    )
    model = ips.models.ApaxEnsemble(models=[model1, model2])
    predictions = ips.analysis.Prediction(model=model, data=test_data)
    metrics = ips.analysis.PredictionMetrics(data=predictions)

    md = ips.calculators.ASEMD(
        data=data.atoms,
        data_id=-1,
        model=model,
        thermostat=thermostat,
        checker_list=[uncertainty_check],
        steps=50000,
        sampling_rate=1,
    )

    selection = ips.configuration_selection.ThresholdSelection(
        data=md.atoms,
        key="forces_uncertainty",
        n_configurations=20,
        dim_reduction="max",
        reduction_axis=(1, 2),
        min_distance=5,
    )

    cp2k = ips.calculators.CP2KSinglePoint(
        data=selection.atoms,
        cp2k_params="config/cp2k.yaml",
        cp2k_files=["GTH_BASIS_SETS", "GTH_POTENTIALS", "dftd3.dat"],
    )

    train_data += cp2k.atoms

with project.group("ML1"):
    model1 = ips.models.Apax(
        data=train_data,
        validation_data=test_data,
        config="config/initial_model_1.yaml",
    )
    model2 = ips.models.Apax(
        data=train_data,
        validation_data=test_data,
        config="config/initial_model_2.yaml",
    )
    model = ips.models.ApaxEnsemble(models=[model1, model2])
    predictions = ips.analysis.Prediction(model=model, data=test_data)
    metrics = ips.analysis.PredictionMetrics(data=predictions)

    md = ips.calculators.ASEMD(
        data=data.atoms,
        data_id=-1,
        model=model,
        thermostat=thermostat,
        checker_list=[uncertainty_check],
        steps=50000,
        sampling_rate=1,
    )

    selection = ips.configuration_selection.ThresholdSelection(
        data=md.atoms,
        key="forces_uncertainty",
        n_configurations=20,
        dim_reduction="max",
        reduction_axis=(1, 2),
        min_distance=5,
    )

    cp2k = ips.calculators.CP2KSinglePoint(
        data=selection.atoms,
        cp2k_params="config/cp2k.yaml",
        cp2k_files=["GTH_BASIS_SETS", "GTH_POTENTIALS", "dftd3.dat"],
    )

    train_data += cp2k.atoms

with project.group("ML2"):
    model1 = ips.models.Apax(
        data=train_data,
        validation_data=test_data,
        config="config/initial_model_1.yaml",
    )
    model2 = ips.models.Apax(
        data=train_data,
        validation_data=test_data,
        config="config/initial_model_2.yaml",
    )
    model = ips.models.ApaxEnsemble(models=[model1, model2])
    predictions = ips.analysis.Prediction(model=model, data=test_data)
    metrics = ips.analysis.PredictionMetrics(data=predictions)

    md = ips.calculators.ASEMD(
        data=md.atoms,
        data_id=-1,
        model=model,
        thermostat=thermostat,
        checker_list=[uncertainty_check],
        steps=50000,
        sampling_rate=1,
    )

    selection = ips.configuration_selection.ThresholdSelection(
        data=md.atoms,
        key="forces_uncertainty",
        n_configurations=50,
        dim_reduction="max",
        reduction_axis=(1, 2),
        min_distance=50,
    )

    cp2k = ips.calculators.CP2KSinglePoint(
        data=selection.atoms,
        cp2k_params="config/cp2k.yaml",
        cp2k_files=["GTH_BASIS_SETS", "GTH_POTENTIALS", "dftd3.dat"],
    )

    train_data += cp2k.atoms

with project.group("ML3"):
    model1 = ips.models.Apax(
        data=train_data,
        validation_data=test_data,
        config="config/initial_model_1.yaml",
    )
    model2 = ips.models.Apax(
        data=train_data,
        validation_data=test_data,
        config="config/initial_model_2.yaml",
    )
    model = ips.models.ApaxEnsemble(models=[model1, model2])
    predictions = ips.analysis.Prediction(model=model, data=test_data)
    metrics = ips.analysis.PredictionMetrics(data=predictions)

    md = ips.calculators.ASEMD(
        data=md.atoms,
        data_id=-1,
        model=model,
        thermostat=thermostat,
        checker_list=[uncertainty_check],
        steps=50000,
        sampling_rate=1,
    )

    selection = ips.configuration_selection.ThresholdSelection(
        data=md.atoms,
        key="forces_uncertainty",
        n_configurations=50,
        dim_reduction="max",
        reduction_axis=(1, 2),
        min_distance=50,
    )

    cp2k = ips.calculators.CP2KSinglePoint(
        data=selection.atoms,
        cp2k_params="config/cp2k.yaml",
        cp2k_files=["GTH_BASIS_SETS", "GTH_POTENTIALS", "dftd3.dat"],
    )

    train_data += cp2k.atoms

with project.group("ML4"):
    model1 = ips.models.Apax(
        data=train_data,
        validation_data=test_data,
        config="config/initial_model_1.yaml",
    )
    model2 = ips.models.Apax(
        data=train_data,
        validation_data=test_data,
        config="config/initial_model_2.yaml",
    )
    model = ips.models.ApaxEnsemble(models=[model1, model2])
    predictions = ips.analysis.Prediction(model=model, data=test_data)
    metrics = ips.analysis.PredictionMetrics(data=predictions)

    md = ips.calculators.ASEMD(
        data=md.atoms,
        data_id=-1,
        model=model,
        thermostat=thermostat,
        checker_list=[uncertainty_check],
        steps=50000,
        sampling_rate=1,
    )

    selection = ips.configuration_selection.ThresholdSelection(
        data=md.atoms,
        key="forces_uncertainty",
        n_configurations=50,
        dim_reduction="max",
        reduction_axis=(1, 2),
        min_distance=50,
    )

    # cp2k = ips.calculators.CP2KSinglePoint(
    #     data=selection.atoms,
    #     cp2k_params="config/cp2k.yaml",
    #     cp2k_files=["GTH_BASIS_SETS", "GTH_POTENTIALS", "dftd3.dat"],
    # )

    # train_data += cp2k.atoms

project.build()

# cp2k = ips.calculators.CP2KSinglePoint(
#     data=None,
#     cp2k_params="config/cp2k.yaml",
#     cp2k_files=["GTH_BASIS_SETS", "GTH_POTENTIALS", "dftd3.dat"],
# )

# mapping = ips.geometry.BarycenterMapping(data=None)

# with project.group("GeoOpt"):
#     cation = ips.configuration_generation.SmilesToAtoms("CCCCN1C=C[N+](=C1)C")
#     anion = ips.configuration_generation.SmilesToAtoms("[B-](F)(F)(F)F")

#     single_structure = ips.configuration_generation.Packmol(
#         data=[cation.atoms, anion.atoms],
#         count=[1, 1],
#         density=1210,  # 293 K
#     )

#     structure = ips.configuration_generation.Packmol(
#         data=[single_structure.atoms],
#         count=[10],
#         density=1210,
#     )

#     geo_opt = ips.calculators.ASEGeoOpt(
#         model=cp2k,
#         data=structure.atoms,
#         optimizer="BFGS",
#         run_kwargs={"fmax": 0.1},
#     )

# mmk_kernel = ips.configuration_comparison.MMKernel(
#         use_jit=True,
#         soap={
#             "r_cut": 6.0,
#             "n_max": 7,
#             "l_max": 5,
#             "sigma": 0.5,
#         },
#     )

# with project.group("bootstrap"):
#     rotate = ips.bootstrap.RotateMolecules(
#         data=geo_opt.atoms,
#         data_id=-1,
#         n_configurations=500,
#         maximum=10 * 3.1415 / 180,  # deg max rotation
#         include_original=False,
#         seed=1,
#     )
#     translate = ips.bootstrap.TranslateMolecules(
#         data=geo_opt.atoms,
#         data_id=-1,
#         n_configurations=500,
#         maximum=0.2,  # Ang max molecular displacement
#         include_original=False,
#         seed=1,
#     )

#     mmk_selection = ips.configuration_selection.KernelSelection(
#             correlation_time=1,
#             n_configurations=50,
#             kernel=mmk_kernel,
#             data=rotate.atoms + translate.atoms,
#             threshold=0.99,
#         )

#     cp2k = ips.calculators.CP2KSinglePoint(
#         data=mmk_selection.atoms,
#         cp2k_params="config/cp2k.yaml",
#         cp2k_files=["GTH_BASIS_SETS", "GTH_POTENTIALS", "dftd3.dat"],
#     )


# with project.group("ML0") as grp:
#     mmk_selection = ips.configuration_selection.KernelSelection(
#             correlation_time=1,
#             n_configurations=50,
#             kernel=mmk_kernel,
#             data=geo_opt.atoms,
#             name="MMK",
#             threshold=0.995,
#         )

#     training_data = mmk_selection.atoms + cp2k.atoms

#     model1 = ips.models.Apax(
#             data=training_data,
#             validation_data=mmk_selection.excluded_atoms,
#             config="config/initial_model_1.yaml"
#         )

#     model2 = ips.models.Apax(
#             data=training_data,
#             validation_data=mmk_selection.excluded_atoms,
#             config="config/initial_model_2.yaml"
#         )

#     model = ips.models.ApaxEnsemble(models=[model1, model2])

#     prediction = ips.analysis.Prediction(model=model, data=mmk_selection.excluded_atoms)
#     metrics = ips.analysis.PredictionMetrics(data=prediction)

# thermostat = ips.calculators.LangevinThermostat(
#     temperature=298.15, friction=0.01, time_step=0.5
# )

# threshold = ips.nodes.ThresholdCheck(
#         value="forces",
#         max_value=2.0,
#     )
# with project.group("MD") as md_grp:
# md = ips.calculators.ASEMD(
#             data=geo_opt.atoms,
#             data_id=-1,
#             model=model,
#             thermostat=thermostat,
#             checker_list=[threshold],
#             steps=50000,
#             sampling_rate=1,
#         )

#     selection = ips.configuration_selection.KernelSelection(
#             correlation_time=1,
#             n_configurations=20,
#             kernel=mmk_kernel,
#             data=md.atoms,
#             initial_configurations=training_data,
#             threshold=0.99,
#         )

#     # cp2k = ips.calculators.CP2KSinglePoint(
#     #     data=selection.atoms,
#     #     cp2k_params="config/cp2k.yaml",
#     #     cp2k_files=["GTH_BASIS_SETS", "GTH_POTENTIALS", "dftd3.dat"],
#     # )

#     # training_data += cp2k.atoms

# project.build()
