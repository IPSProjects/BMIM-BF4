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

with project.group("ML0") as grp:
    seed_configs = ips.configuration_selection.RandomSelection(
            data=geo_opt.atoms, n_configurations=16, seed=1, name="seed"
        )

    model = ips.models.Apax(
            data=seed_configs.atoms,
            validation_data=seed_configs.excluded_atoms,
            config="config/initial_model_1.yaml"
        )
    
    prediction = ips.analysis.Prediction(model=model, data=seed_configs.excluded_atoms)
    metrics = ips.analysis.PredictionMetrics(data=prediction)
    


project.build()
