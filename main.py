import ipsuite as ips
from ase import units

project = ips.Project(automatic_node_names=True)


with project.group("final") as final:
    data = ips.AddData("data/traj.xyz")
    train_selection = ips.configuration_selection.RandomSelection(data=data, n_configurations=450)

    model = ips.models.Apax(config="config/final.yaml", data=train_selection.atoms, validation_data=train_selection.excluded_atoms)
    prediction = ips.analysis.Prediction(data=train_selection.excluded_atoms, model=model)
    metrics = ips.analysis.PredictionMetrics(data=prediction)

project.build()