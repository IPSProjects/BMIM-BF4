import ipsuite as ips
from ase import units

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

for idx in range(8, 11):
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


with project.group(f"ML11_MultiPack") as grp_ml11:
    cation = ips.configuration_generation.SmilesToConformers(
        smiles="CCCCN1C=C[N+](=C1)C", numConfs=200
    )
    anion = ips.configuration_generation.SmilesToConformers(
        smiles="[B-](F)(F)(F)F", numConfs=200
    )
    single_structure = ips.configuration_generation.MultiPackmol(
        data=[cation.atoms, anion.atoms],
        count=[1, 1],
        density=1210,
        pbc=False,
        n_configurations=200,
    )

    structure = ips.configuration_generation.MultiPackmol(
        data=[single_structure.atoms],
        count=[10],
        density=900,
        n_configurations=20,
    )

    cp2k = ips.calculators.CP2KSinglePoint(
        data=structure.atoms,
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


with project.group(f"ML12") as grp:
    md = ips.calculators.ASEMD(
        data=geo_opt.atoms,
        data_id=-1,
        model=model,
        thermostat=thermostat,
        modifier=[eq_box_oszillator, temperature_oszillator],
        checker_list=[uncertainty_check],
        steps=50000,
        sampling_rate=10,
    )
    md_nvt = md
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


mapping = ips.geometry.BarycenterMapping(data=None)

with project.group("VS") as vs:
    vs_list = []
    for idx in range(5):
        geo_opt = ips.calculators.ASEGeoOpt(
            model=model,
            data=structure.atoms,
            data_id=idx,
            optimizer="FIRE",
            run_kwargs={"fmax": 0.5},
        )

        volume_scan = ips.analysis.BoxScale(
            data=geo_opt.atoms,
            mapping=mapping,
            model=model,
            start=0.90,
            stop=2.0,
            num=100,
            data_id=-1,
        )

        vs_list.append(volume_scan.atoms)

    selection = ips.configuration_selection.RandomSelection(
        data=sum(vs_list, []),
        n_configurations=50,
    )

    cp2k = ips.calculators.CP2KSinglePoint(
        data=selection.atoms,
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

thermostat = ips.calculators.NPTThermostat(
        time_step=1.0,
        temperature=300,
        pressure=6.324e-07, # 1.01325 * units.bar,
        ttime=2.4557, # 25 * units.fs,
        pfactor=54.273, # (75 * units.fs) ** 2,
        tetragonal_strain=True,
    )

with project.group("ML13") as grp_ml13:
    md = ips.calculators.ASEMD(
        data=geo_opt.atoms,
        data_id=-1,
        model=model,
        thermostat=thermostat,
        checker_list=[uncertainty_check],
        steps=100000,
        sampling_rate=10,
    )

    kernel_selection = ips.models.apax.BatchKernelSelection(
        data=md.atoms,
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

temperature_oszillator = ips.calculators.TemperatureOscillatingRampModifier(
    end_temperature=500,  # decomp ~ 290
    start_temperature=300,  # melting -75
    num_oscillations=10,
    temperature_amplitude=150,
)

with project.group("ML14") as grp:
    md = ips.calculators.ASEMD(
        data=geo_opt.atoms,
        data_id=-1,
        model=model,
        modifier=[temperature_oszillator],
        thermostat=thermostat,
        checker_list=[uncertainty_check],
        steps=100000,
        sampling_rate=10,
    )

    kernel_selection = ips.models.apax.BatchKernelSelection(
        data=md.atoms,
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



thermostat = ips.calculators.NPTThermostat(
        time_step=1.0,
        temperature=300,
        pressure=6.324e-07, # 1.01325 * units.bar,
        ttime=2.4557, # 25 * units.fs,
        pfactor=54.273, # (75 * units.fs) ** 2,
        tetragonal_strain=True,
    )

temperature_oszillator = ips.calculators.TemperatureOscillatingRampModifier(
    end_temperature=300,  # decomp ~ 290
    start_temperature=200,  # melting -75
    num_oscillations=10,
    temperature_amplitude=100,
)

with project.group("ML15") as grp:
    md_15 = ips.calculators.ASEMD(
        data=md.atoms,
        data_id=-1,
        model=model,
        modifier=[temperature_oszillator],
        thermostat=thermostat,
        checker_list=[uncertainty_check],
        steps=1000000,
        sampling_rate=100,
    )

    kernel_selection = ips.models.apax.BatchKernelSelection(
        data=md_15.atoms,
        train_data=train_data,
        models=model,
        n_configurations=100,
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


with project.group("final") as final:
    model_d3cp2k = ips.models.Apax(
        data=train_data,
        validation_data=validation_data.atoms,
        config="config/final.yaml",
    )
    prediction = ips.analysis.Prediction(data=test_data, model=model_d3cp2k)
    metrics = ips.analysis.PredictionMetrics(data=prediction)

with project.group("final_ensemble") as final_ensemble:
    model = ips.models.Apax(
        data=train_data,
        validation_data=validation_data.atoms,
        config="config/final_ensemble.yaml",
    )
    prediction = ips.analysis.Prediction(data=test_data, model=model)
    metrics = ips.analysis.PredictionMetrics(data=prediction)

    prediction = ips.analysis.Prediction(data=train_data, model=model)
    metrics = ips.analysis.PredictionMetrics(data=prediction)

with project.group("wo_d3") as wo_d3:
    train_data_nod3 = ips.calculators.CP2KSinglePoint(
        data=train_data,
        cp2k_params="config/cp2k_wo_d3.yaml",
        cp2k_files=["GTH_BASIS_SETS", "GTH_POTENTIALS"],
    ).atoms

    validation_data_nod3 = ips.calculators.CP2KSinglePoint(
        data=validation_data.atoms,
        cp2k_params="config/cp2k_wo_d3.yaml",
        cp2k_files=["GTH_BASIS_SETS", "GTH_POTENTIALS"],
    ).atoms


with project.group("wo_d3_model") as wo_d3_model:

    model_wod3 = ips.models.Apax(
        data=train_data_nod3,
        validation_data=validation_data_nod3,
        config="config/final.yaml",
    )


with project.group("torch_d3_labels") as td3l:
    train_d3_short = ips.calculators.TorchD3(
        data=train_data_nod3,
        xc="b97-3c",
        damping="bj",
        cutoff=7.93766,
        cnthr=7.93766,
        abc=False,
        dtype="float32",
    )
    val_d3_short = ips.calculators.TorchD3(
        data=validation_data_nod3,
        xc="b97-3c",
        damping="bj",
        cutoff=7.93766,
        cnthr=7.93766,
        abc=False,
        dtype="float32",
    )
    train_d3_long = ips.calculators.TorchD3(
        data=train_data_nod3,
        xc="b97-3c",
        damping="bj",
        cutoff=40.0,
        cnthr=20.0,
        abc=False,
        dtype="float32",
    )
    val_d3_long = ips.calculators.TorchD3(
        data=validation_data_nod3,
        xc="b97-3c",
        damping="bj",
        cutoff=40.0,
        cnthr=20.0,
        abc=False,
        dtype="float32",
    )


with project.group("d3_models") as d3_models:

    model_short = ips.models.Apax(
        data=train_d3_short,
        validation_data=val_d3_short,
        config="config/final.yaml",
    )

    model_long = ips.models.Apax(
        data=train_d3_long,
        validation_data=val_d3_long,
        config="config/final.yaml",
    )


with project.group("ML16") as grp:

    geo_opt = ips.calculators.ASEGeoOpt(
        model=model_wod3,
        data=md_15.atoms,
        data_id=-1,
        optimizer="FIRE",
        run_kwargs={"fmax": 0.5},
    )

    kernel_selection = ips.models.apax.BatchKernelSelection(
        data=geo_opt.atoms,
        train_data=train_data_nod3,
        models=model_wod3,
        n_configurations=5,
        processing_batch_size=4,
    )

    md = ips.calculators.ASEMD(
        data=geo_opt.atoms,
        data_id=-1,
        model=model_wod3,
        modifier=[],
        thermostat=thermostat,
        checker_list=[],
        steps=1_000_000,
        sampling_rate=100,
    )

    validation_selection = ips.configuration_selection.RandomSelection(
        data=md.atoms, n_configurations=100
    )

    train_selection = ips.configuration_selection.RandomSelection(
        data=validation_selection.excluded_atoms, n_configurations=100
    )

    cp2k_train = ips.calculators.CP2KSinglePoint(
        data=train_selection.atoms + kernel_selection.atoms,
        cp2k_params="config/cp2k_wo_d3.yaml",
        cp2k_files=["GTH_BASIS_SETS", "GTH_POTENTIALS"],
    )
    cp2k_val = ips.calculators.CP2KSinglePoint(
        data=validation_selection.atoms,
        cp2k_params="config/cp2k_wo_d3.yaml",
        cp2k_files=["GTH_BASIS_SETS", "GTH_POTENTIALS"],
    )
    train_data_nod3 += cp2k_train.atoms
    validation_data_nod3 += cp2k_val.atoms

    model = ips.models.Apax(
        data=train_data_nod3,
        validation_data=validation_data_nod3,
        config="config/wod3.yaml",
    )

    prediction = ips.analysis.Prediction(data=validation_data_nod3, model=model)
    metrics = ips.analysis.PredictionMetrics(data=prediction)
    ips.analysis.EnergyHistogram(data=train_data_nod3, bins=100)
    ips.analysis.ForcesHistogram(data=train_data_nod3)
    ips.analysis.ForceDecomposition(data=prediction)


thermostat = ips.calculators.LangevinThermostat(
    temperature=298.15, friction=0.01, time_step=0.5
)
temperature_oszillator = ips.calculators.TemperatureOscillatingRampModifier(
    end_temperature=500,  # decomp ~ 290
    start_temperature=230,  # melting -75
    num_oscillations=5,
    temperature_amplitude=100,
)
barostat = ips.calculators.NPTThermostat(
    time_step=0.5,
    temperature=298,
    pressure=6.324e-07, # 1.01325 * units.bar,
    ttime=2.4557, # 25 * units.fs,
    pfactor=54.273, # (75 * units.fs) ** 2,
    tetragonal_strain=True,
)

with project.group("ML17_sampling") as grp:
    geo_opt = ips.calculators.ASEGeoOpt(
        model=model,
        data=md_nvt.atoms, # latest configuration at experimental density
        data_id=-1,
        optimizer="FIRE",
        run_kwargs={"fmax": 0.5},
    )

    md_tempramp = ips.calculators.ASEMD(
        data=geo_opt.atoms,
        data_id=-1,
        model=model,
        modifier=[temperature_oszillator],
        thermostat=thermostat,
        checker_list=[],
        steps=1_000_000,
        sampling_rate=100,
    )

    td3 = ips.calculators.TorchD3(
        data=validation_data_nod3,
        xc="b97-3c",
        damping="bj",
        cutoff=7.93766,
        cnthr=7.93766,
        abc=False,
        dtype="float32",
        skin=0.5,
    )

    model_td3_mix = ips.calculators.MixCalculator(
        data=validation_data_nod3,
        calculators=[model, td3],
        methods="sum",
    )

    md_npt = ips.calculators.ASEMD(
        data=geo_opt.atoms,
        data_id=-1,
        model=model_td3_mix,
        modifier=[],
        thermostat=barostat,
        checker_list=[],
        steps=1_000_000,
        sampling_rate=100,
    )

    train_selection_opt = ips.models.apax.BatchKernelSelection(
        data=geo_opt.atoms,
        train_data=train_data_nod3,
        models=model,
        n_configurations=4,
        processing_batch_size=4,
    )

    val_selection_nvt = ips.configuration_selection.RandomSelection(
        data=md_tempramp.atoms, n_configurations=40
    )
    train_selection_nvt = ips.configuration_selection.RandomSelection(
        data=val_selection_nvt.excluded_atoms, n_configurations=80
    )

    val_selection_npt = ips.configuration_selection.RandomSelection(
        data=md_npt.atoms, n_configurations=40
    )
    train_selection_npt = ips.configuration_selection.RandomSelection(
        data=val_selection_npt.excluded_atoms, n_configurations=80
    )

with project.group("ML17_datasets") as ml17_data:
    cp2k_train = ips.calculators.CP2KSinglePoint(
        data=train_selection_opt.atoms + train_selection_nvt.atoms + train_selection_npt.atoms,
        cp2k_params="config/cp2k_wo_d3.yaml",
        cp2k_files=["GTH_BASIS_SETS", "GTH_POTENTIALS"],
    )
    cp2k_val = ips.calculators.CP2KSinglePoint(
        data=val_selection_nvt.atoms + val_selection_npt.atoms,
        cp2k_params="config/cp2k_wo_d3.yaml",
        cp2k_files=["GTH_BASIS_SETS", "GTH_POTENTIALS"],
    )
    train_data_nod3 += cp2k_train.atoms
    validation_data_nod3 += cp2k_val.atoms


    train_d3_short = ips.calculators.TorchD3(
        data=train_data_nod3,
        xc="b97-3c",
        damping="bj",
        cutoff=7.93766,
        cnthr=7.93766,
        abc=False,
        dtype="float32",
    )
    val_d3_short = ips.calculators.TorchD3(
        data=validation_data_nod3,
        xc="b97-3c",
        damping="bj",
        cutoff=7.93766,
        cnthr=7.93766,
        abc=False,
        dtype="float32",
    )
    train_d3_medium = ips.calculators.TorchD3(
        data=train_data_nod3,
        xc="b97-3c",
        damping="bj",
        cutoff=20.0,
        cnthr=20.0,
        abc=False,
        dtype="float32",
    )
    val_d3_medium = ips.calculators.TorchD3(
        data=validation_data_nod3,
        xc="b97-3c",
        damping="bj",
        cutoff=20.0,
        cnthr=20.0,
        abc=False,
        dtype="float32",
    )


with project.group("ML17_training") as ml17_train:
    model_nod3 = ips.models.Apax(
        data=train_data_nod3,
        validation_data=validation_data_nod3,
        config="config/ml17_ensemble.yaml",
    )
    model_short = ips.models.Apax(
        data=train_d3_short,
        validation_data=val_d3_short,
        config="config/ml17_ensemble.yaml",
    )
    model_long_cutoff = ips.models.Apax(
        data=train_d3_short,
        validation_data=val_d3_short,
        config="config/ml17_ensemble_long_cutoff.yaml",
    )
    model_long = ips.models.Apax(
        data=train_d3_medium,
        validation_data=val_d3_medium,
        config="config/ml17_ensemble.yaml",
    )


with project.group("ML17_eval") as ml17_eval:
    ips.analysis.EnergyHistogram(data=train_data_nod3, bins=100)
    ips.analysis.ForcesHistogram(data=train_data_nod3)
    ips.analysis.EnergyHistogram(data=train_d3_short, bins=100)
    ips.analysis.ForcesHistogram(data=train_d3_short)
    ips.analysis.EnergyHistogram(data=train_d3_medium, bins=100)
    ips.analysis.ForcesHistogram(data=train_d3_medium)

    prediction = ips.analysis.Prediction(data=validation_data_nod3, model=model_nod3)
    metrics = ips.analysis.PredictionMetrics(data=prediction)
    ips.analysis.ForceDecomposition(data=prediction)

    prediction = ips.analysis.Prediction(data=val_d3_short, model=model_short)
    metrics = ips.analysis.PredictionMetrics(data=prediction)
    ips.analysis.ForceDecomposition(data=prediction)

    prediction = ips.analysis.Prediction(data=val_d3_short, model=model_long_cutoff)
    metrics = ips.analysis.PredictionMetrics(data=prediction)
    ips.analysis.ForceDecomposition(data=prediction)

    prediction = ips.analysis.Prediction(data=val_d3_medium, model=model_long)
    metrics = ips.analysis.PredictionMetrics(data=prediction)
    ips.analysis.ForceDecomposition(data=prediction)

with project.group("ml17_mace") as ml17_mace:
    mace = ips.models.MACE(
        data=train_data_nod3,
        test_data=validation_data_nod3,
        config="config/mace_foundation.yaml",
    )
    pred = ips.analysis.Prediction(data=validation_data_nod3, model=mace)
    metrics = ips.analysis.PredictionMetrics(data=pred)
    
project.build(nodes=[ml17_mace])
