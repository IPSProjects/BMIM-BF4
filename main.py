import ipsuite as ips

project = ips.Project(automatic_node_names=True)
cp2k_shell = "mpirun -n 32 cp2k_shell.psmp"

with project:
    data = ips.data_loading.AddData("data/BMIM_BF4_400_00K.extxyz")
    confs = ips.configuration_selection.RandomSelection(data=data.atoms, n_configurations=10)
    cp2k = ips.calculators.CP2KSinglePoint(
        data=confs.atoms,
        cp2k_params="config/cp2k.yaml",
        cp2k_files=["GTH_BASIS_SETS", "GTH_POTENTIALS", "dftd3.dat"],
        cp2k_shell=cp2k_shell,
    )

project.build()