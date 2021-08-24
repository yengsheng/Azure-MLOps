from azureml.core import Workspace, Datastore, Dataset
from ml_service.util.env_variables import Env

def main():
    e = Env()

    ws = Workspace.get(
        name=e.workspace_name,
        subscription_id=e.subscription_id,
        resource_group=e.resource_group,
    )
    print(ws.name, 'loaded.')
    default_ds = ws.get_default_datastore()
    directory = './data/'
    default_ds.upload_files(files=['./data/aviation_main.csv',
                                    './data/aviation_inflow.csv'],
                        target_path = 'aviation-data/',
                        overwrite = True,
                        show_progress = True)
    csv_paths = [(default_ds, 'aviation-data/aviation_inflow.csv')]
    tab_ds = Dataset.Tabular.from_delimited_files(path=csv_paths)
    tab_ds = tab_ds.register(workspace=ws, name='inflow')
    
    csv_paths = [(default_ds, 'aviation-data/aviation_main.csv')]
    tab_ds = Dataset.Tabular.from_delimited_files(path=csv_paths)
    tab_ds = tab_ds.register(workspace=ws, name='main')

if __name__ == '__main__':
    main()
    
# def get_ws():
#     ws = Workspace.from_config()
#     print(ws.name, 'loaded.')
#     return ws

# def get_datastore():
#     default_ds = ws.get_default_datastore()
#     directory = './data/'

# default_ds.upload_file(file='./data/aviation_main.csv',
#                         target_path = 'aviation-data/',
#                         overwrite = True,
#                         show_progress = True)

