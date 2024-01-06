import ipsuite as ips
from ase import units
import mace_models

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

thermostat = ips.calculators.NPTThermostat(
        time_step=0.5,
        temperature=303,
        pressure=1.01325 * units.bar,
        ttime=25 * units.fs,
        pfactor=(75 * units.fs) ** 2,
        tetragonal_strain=True,
    )

with project.group("AIMD") as aimd_grp:
    start_conf = ips.configuration_selection.IndexSelection(
        data.atoms,
        indices=[2000,]
    )

    aimd_model = ips.calculators.CP2KSinglePoint(
        data=start_conf.atoms,
        cp2k_params="config/cp2k.yaml",
        cp2k_files=["GTH_BASIS_SETS", "GTH_POTENTIALS", "dftd3.dat"],
    )

    aimd = ips.calculators.ASEMD(
        data=start_conf.atoms,
        data_id=-1,
        model=aimd_model,
        thermostat=thermostat,
        steps=13_200,
        sampling_rate=1,
        dump_rate=100,
    )

    density = ips.analysis.AnalyseDensity(data=aimd.atoms)

mace_mp_0_model = mace_models.LoadModel.from_rev("MACE-MP-0", rev="MACE-MP-0", remote="https://github.com/RokasEl/MACE-Models.git")

with project.group("density_md", "MACE-MP-0") as mace_mp_0:
    mace_md = ips.calculators.ASEMD(
        data=start_conf.atoms,
        data_id=-1,
        model=mace_mp_0_model,
        thermostat=thermostat,
        steps=20_000,
        sampling_rate=1,
        dump_rate=100,
    )

    density = ips.analysis.AnalyseDensity(data=mace_md.atoms)

mace_d3 = ips.calculators.TorchD3(
    data=None,
    xc="pbe",
    damping="bj",
    cutoff=20.0,
    cnthr=18.0,
    abc=False,
    dtype="float32",
)


with project.group("density_md", "MACE-MP-0-d3") as mace_mp_0:
    mace_mp_0_model_with_d3 = ips.calculators.MixCalculator(data=start_conf.atoms, calculators=[mace_mp_0_model, mace_d3])

    mace_md = ips.calculators.ASEMD(
        data=start_conf.atoms,
        data_id=-1,
        model=mace_mp_0_model_with_d3,
        thermostat=thermostat,
        steps=20_000,
        sampling_rate=1,
        dump_rate=100,
    )

    density = ips.analysis.AnalyseDensity(data=mace_md.atoms)

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

with project.group("final") as final:
    model = ips.models.Apax(
        data=train_data,
        validation_data=validation_data.atoms,
        config="config/final.yaml",
    )
    prediction = ips.analysis.Prediction(data=test_data, model=model)
    metrics = ips.analysis.PredictionMetrics(data=prediction)

with project.group("final_ensemble") as final:
    model = ips.models.Apax(
        data=train_data,
        validation_data=validation_data.atoms,
        config="config/final_ensemble.yaml",
    )
    prediction = ips.analysis.Prediction(data=test_data, model=model)
    metrics = ips.analysis.PredictionMetrics(data=prediction)

    prediction = ips.analysis.Prediction(data=train_data, model=model)
    metrics = ips.analysis.PredictionMetrics(data=prediction)

project.build(nodes=[mace_mp_0])
