from azureml.core import Workspace, Datastore, Dataset
from ml_service.util.env_variables import Env
import pandas as pd

def main():
    e = Env()

    ws = Workspace.get(
        name=e.workspace_name,
        subscription_id=e.subscription_id,
        resource_group=e.resource_group,
    )
    print(ws.name, 'loaded.')
    default_ds = ws.get_default_datastore()
    # directory = './data/'
    # default_ds.upload_files(files=['./data/aviation_main.csv',
    #                                 './data/aviation_inflow.csv'],
    #                     target_path = 'aviation-data/',
    #                     overwrite = True,
    #                     show_progress = True)
    # csv_paths = [(default_ds, 'aviation-data/aviation_inflow.csv')]
    # tab_ds = Dataset.Tabular.from_delimited_files(path=csv_paths)
    # tab_ds = tab_ds.register(workspace=ws, name='inflow')
    
    # csv_paths = [(default_ds, 'aviation-data/aviation_main.csv')]
    # tab_ds = Dataset.Tabular.from_delimited_files(path=csv_paths)
    # tab_ds = tab_ds.register(workspace=ws, name='main')

    main_ds = Dataset.get_by_name(ws, name='main').to_pandas_dataframe()
    inflow_ds = Dataset.get_by_name(ws, name='inflow').to_pandas_dataframe()

    new_main = pd.concat([main_ds.tail(len(main_ds) - 1000), inflow_ds.head(1000)], ignore_index = True)
    new_inflow = pd.concat([inflow_ds.tail(len(inflow_ds) - 1000), main_ds.head(1000)], ignore_index = True)

    new_main = Dataset.Tabular.register_pandas_dataframe(new_main, target = default_ds, name = 'main', show_progress = True)
    new_inflow = Dataset.Tabular.register_pandas_dataframe(new_inflow, target = default_ds, name = 'inflow', show_progress = True)

if __name__ == '__main__':
    main()