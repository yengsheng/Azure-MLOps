from azureml.core import Workspace
from azureml.core.model import Model
from ml_service.util.env_variables import Env

def main():
    e = Env()
    model_name = 'aviation_model'

    ws = Workspace.get(
        name=e.workspace_name,
        subscription_id=e.subscription_id,
        resource_group=e.resource_group,
    )
    print(ws.name, 'loaded.')
    model = Model(ws, model_name)
    return model.version

if __name__ == '__main__':
    main()