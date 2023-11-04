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

mmk_kernel = ips.configuration_comparison.MMKernel(
        use_jit=True,
        soap={
            "r_cut": 6.0,
            "n_max": 7,
            "l_max": 5,
            "sigma": 0.5,
        },
    )

with project.group("bootstrap"):
    rotate = ips.bootstrap.RotateMolecules(
        data=geo_opt.atoms,
        data_id=-1,
        n_configurations=500,
        maximum=10 * 3.1415 / 180,  # deg max rotation
        include_original=False,
        seed=1,
    )
    translate = ips.bootstrap.TranslateMolecules(
        data=geo_opt.atoms,
        data_id=-1,
        n_configurations=500,
        maximum=0.2,  # Ang max molecular displacement
        include_original=False,
        seed=1,
    )

    mmk_selection = ips.configuration_selection.KernelSelection(
            correlation_time=1,
            n_configurations=50,
            kernel=mmk_kernel,
            data=rotate.atoms + translate.atoms,
            threshold=0.99,
        )


with project.group("ML0") as grp:
    mmk_selection = ips.configuration_selection.KernelSelection(
            correlation_time=1,
            n_configurations=20,
            kernel=mmk_kernel,
            data=geo_opt.atoms,
            name="MMK",
            threshold=0.99,
        )
    
    model1 = ips.models.Apax(
            data=mmk_selection.atoms,
            validation_data=mmk_selection.excluded_atoms,
            config="config/initial_model_1.yaml"
        )
    
    model2 = ips.models.Apax(
            data=mmk_selection.atoms,
            validation_data=mmk_selection.excluded_atoms,
            config="config/initial_model_2.yaml"
        )

    model = ips.models.ApaxEnsemble(models=[model1, model2])

    prediction = ips.analysis.Prediction(model=model, data=mmk_selection.excluded_atoms)
    metrics = ips.analysis.PredictionMetrics(data=prediction)

# thermostat = ips.calculators.LangevinThermostat(
#     temperature=298.15, friction=0.01, time_step=0.5
# )

# threshold = ips.nodes.ThresholdCheck(
#         value="forces",
#         max_value=0.5,
#     )
# with project.group("MD") as md_grp:
#     md = ips.calculators.ASEMD(
#                 data=geo_opt.atoms,
#                 data_id=-1,
#                 model=model,
#                 thermostat=thermostat,
#                 checker_list=[threshold],
#                 steps=50000,
#                 sampling_rate=1,
#             )
project.build()
