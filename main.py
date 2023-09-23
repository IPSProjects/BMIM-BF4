import ipsuite as ips

project = ips.Project(automatic_node_names=True)
cp2k_shell = "mpirun -n 8 cp2k_shell.psmp"

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
    train_data_bootstrap = ips.configuration_selection.RandomSelection(
        data=data, n_configurations=30
    )
    geoopt_data = ips.configuration_selection.IndexSelection(
        data=geopt.atoms, indices=slice(10, None, None)
    )

    train_data_geoopt = ips.configuration_selection.RandomSelection(
        data=geoopt_data, n_configurations=50
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

    model = ips.models.EnsembleModel(models=[model_1, model_2])

    prediction = ips.analysis.Prediction(data=validation_data, model=model)
    metrics = ips.analysis.PredictionMetrics(data=prediction)


thermostat = ips.calculators.LangevinThermostat(
    temperature=298.15, friction=0.01, time_step=0.5
)

uncertainty_check = ips.analysis.ThresholdCheck(
    value="energy_uncertainty", max_value=2.0, larger_only=True
)

models = []

for loop in range(5):
    with project.group(f"AL{loop}", "train") as altrain:
        md = ips.calculators.ASEMD(
            data=geopt.atoms if loop == 0 else ref_geopt_train.atoms,
            data_id=-1,
            model=model,
            thermostat=thermostat,
            checker_list=[uncertainty_check],
            steps=50000,
            sampling_rate=1,
        )

        model_geopt = ips.calculators.ASEGeoOpt(
            model=model,
            data=md.atoms,
            data_id=-5,
            optimizer="BFGS",
            run_kwargs={"fmax": 1.0},
            checker_list=[uncertainty_check],
        )

        ref_geopt_train = ips.calculators.ASEGeoOpt(
            model=cp2k_rotate,  # better restart wavefunction
            data=model_geopt.atoms,
            data_id=-1,
            optimizer="BFGS",
            run_kwargs={"fmax": 2.0},
        )

        md_selection = ips.configuration_selection.IndexSelection(
            data=md.atoms, indices=slice(0, -5)
        )
        confs = ips.configuration_selection.RandomSelection(
            data=md_selection.atoms, n_configurations=10
        )

        cp2k = ips.calculators.CP2KSinglePoint(
            data=confs.atoms,
            cp2k_params="config/cp2k.yaml",
            cp2k_files=["GTH_BASIS_SETS", "GTH_POTENTIALS", "dftd3.dat"],
            cp2k_shell=cp2k_shell,
        )

        train_data += cp2k.atoms
        train_data += ref_geopt_train.atoms

    with project.group(f"AL{loop}", "test") as altest:
        md = ips.calculators.ASEMD(
            data=geopt.atoms if loop == 0 else ref_geopt_test.atoms,
            data_id=-10 if loop == 0 else -1,  # slightly different start configuration
            model=model,
            thermostat=thermostat,
            checker_list=[uncertainty_check],
            steps=50000,
            sampling_rate=1,
        )

        model_geopt = ips.calculators.ASEGeoOpt(
            model=model,
            data=md.atoms,
            data_id=-5,
            optimizer="BFGS",
            run_kwargs={"fmax": 1.0},
            checker_list=[uncertainty_check],
        )

        ref_geopt_test = ips.calculators.ASEGeoOpt(
            model=cp2k_rotate,  # better restart wavefunction
            data=model_geopt.atoms,
            data_id=-1,
            optimizer="BFGS",
            run_kwargs={"fmax": 2.0},
        )

        md_selection = ips.configuration_selection.IndexSelection(
            data=md.atoms, indices=slice(0, -5)
        )
        confs = ips.configuration_selection.RandomSelection(
            data=md_selection.atoms, n_configurations=3
        )

        cp2k = ips.calculators.CP2KSinglePoint(
            data=confs.atoms,
            cp2k_params="config/cp2k.yaml",
            cp2k_files=["GTH_BASIS_SETS", "GTH_POTENTIALS", "dftd3.dat"],
            cp2k_shell=cp2k_shell,
        )

        validation_data += ref_geopt_test.atoms
        validation_data += cp2k.atoms

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

        model = ips.models.EnsembleModel(models=[model_1, model_2])


eq_box_oszillator = ips.calculators.BoxOscillatingRampModifier(
    end_cell=14.5839,
    cell_amplitude=1,
    num_oscillations=3,
)

temperature_oszillator = ips.calculators.TemperatureOscillatingRampModifier(
    end_temperature=500,  # boiling around 460
    start_temperature=270,  # melting around 290
    num_oscillations=10,
    temperature_amplitude=150,
)

for loop in range(5, 10):
    with project.group(f"AL{loop}", "train") as altrain:
        md = ips.calculators.ASEMD(
            data=geopt.atoms if loop == 0 else ref_geopt_train.atoms,
            data_id=-1,
            model=model,
            thermostat=thermostat,
            checker_list=[uncertainty_check],
            modifier=[temperature_oszillator, eq_box_oszillator],
            steps=1_000_000,
            sampling_rate=50,
        )

        model_geopt = ips.calculators.ASEGeoOpt(
            model=model,
            data=md.atoms,
            data_id=-5,
            optimizer="BFGS",
            run_kwargs={"fmax": 1.0},
            checker_list=[uncertainty_check],
        )

        ref_geopt_train = ips.calculators.ASEGeoOpt(
            model=cp2k_rotate,  # better restart wavefunction
            data=model_geopt.atoms,
            data_id=-1,
            optimizer="BFGS",
            run_kwargs={"fmax": 2.0},
        )

        md_selection = ips.configuration_selection.IndexSelection(
            data=md.atoms, indices=slice(0, -5)
        )
        confs = ips.configuration_selection.RandomSelection(
            data=md_selection.atoms, n_configurations=10
        )

        cp2k = ips.calculators.CP2KSinglePoint(
            data=confs.atoms,
            cp2k_params="config/cp2k.yaml",
            cp2k_files=["GTH_BASIS_SETS", "GTH_POTENTIALS", "dftd3.dat"],
            cp2k_shell=cp2k_shell,
        )

        train_data += cp2k.atoms
        train_data += ref_geopt_train.atoms

    with project.group(f"AL{loop}", "test") as altest:
        md = ips.calculators.ASEMD(
            data=geopt.atoms if loop == 0 else ref_geopt_test.atoms,
            data_id=-10 if loop == 0 else -1,  # slightly different start configuration
            model=model,
            thermostat=thermostat,
            checker_list=[uncertainty_check],
            modifier=[temperature_oszillator, eq_box_oszillator],
            steps=1_000_000,
            sampling_rate=50,
        )

        model_geopt = ips.calculators.ASEGeoOpt(
            model=model,
            data=md.atoms,
            data_id=-5,
            optimizer="BFGS",
            run_kwargs={"fmax": 1.0},
            checker_list=[uncertainty_check],
        )

        ref_geopt_test = ips.calculators.ASEGeoOpt(
            model=cp2k_rotate,  # better restart wavefunction
            data=model_geopt.atoms,
            data_id=-1,
            optimizer="BFGS",
            run_kwargs={"fmax": 2.0},
        )

        md_selection = ips.configuration_selection.IndexSelection(
            data=md.atoms, indices=slice(0, -5)
        )
        confs = ips.configuration_selection.RandomSelection(
            data=md_selection.atoms, n_configurations=3
        )

        cp2k = ips.calculators.CP2KSinglePoint(
            data=confs.atoms,
            cp2k_params="config/cp2k.yaml",
            cp2k_files=["GTH_BASIS_SETS", "GTH_POTENTIALS", "dftd3.dat"],
            cp2k_shell=cp2k_shell,
        )

        validation_data += ref_geopt_test.atoms
        validation_data += cp2k.atoms

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

        model = ips.models.EnsembleModel(models=[model_1, model_2])

with project.group("metrics") as metrics:
    prediction = ips.analysis.Prediction(data=validation_data, model=model)
    metrics = ips.analysis.PredictionMetrics(data=prediction)


locality_constraints = {}
for atom_type in ["N", "C", "H", "F"]:
    locality_constraints[atom_type] = []
    for radius in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        locality_constraints[atom_type].append(
            ips.calculators.FixedSphereConstraint(
                atom_id=0, radius=radius, atom_type=atom_type
            )
        )

locality_groups = []
for atom_type in locality_constraints:
    with project.group("locality", atom_type) as group:
        locality_groups.append(group)
        md_list = []
        dft_list = []
        for constraint in locality_constraints[atom_type]:
            md = ips.calculators.ASEMD(
                    data=geopt.atoms,
                    data_id=-1,
                    model=model,
                    thermostat=thermostat,
                    checker_list=[uncertainty_check],
                    constraint_list=[constraint],
                    steps=7000,
                    sampling_rate=1000,
                )
            md_list.append(md)

            cp2k = ips.calculators.CP2KSinglePoint(
                data=md.atoms,
                cp2k_params="config/cp2k.yaml",
                cp2k_files=["GTH_BASIS_SETS", "GTH_POTENTIALS", "dftd3.dat"],
                cp2k_shell=cp2k_shell,
            )
            dft_list.append(cp2k)
        
        analysis = ips.analysis.AnalyseSingleForceSensitivity(
            data=dft_list,
            sim_list=md_list,
        )

project.build(nodes=locality_groups)
