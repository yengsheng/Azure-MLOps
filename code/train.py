from azureml.core import Workspace, Dataset
from ml_service.util.env_variables import Env
from azureml.core import Experiment
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
import os
from azureml.core.model import Model
import argparse

def main():

    parser = argparse.ArgumentParser("register")
    parser.add_argument(
        "--output_new_register_file",
        type=str,
        default="new_registration.txt",
        help="Name of a file to write whether to register new model or not"
    )
    args = parser.parse_args()

    e = Env()
    model_name = 'aviation_model'

    ws = Workspace.get(
        name=e.workspace_name,
        subscription_id=e.subscription_id,
        resource_group=e.resource_group,
    )
    print(ws.name, 'loaded.')

    experiment = Experiment(workspace=ws, name="aviation-experiment")
    run = experiment.start_logging()
    print("Starting experiment:", experiment.name)

    #Retrieving current model and save accuracy, if first run, then production_accuracy is set to 0.
    try:
        model = ws.models[model_name]
        production_accuracy = float(model.properties['Accuracy'])
        print(production_accuracy)
    except KeyError:
        production_accuracy = 0

    # Load dataset
    dataset = Dataset.get_by_name(ws, name='main').to_pandas_dataframe()
    # Dataset loaded

    # Separate features and labels
    X, y = dataset[['Investigation_Type','Country','Injury_Severity','Amateur_Built','Number_of_Engines','Total_Fatal_Injuries','Total_Serious_Injuries','Total_Minor_Injuries', 'Total_Uninjured']].values, dataset['Aircraft_damage'].values

    # Split data into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

    # Train a decision tree model
    print('Training a decision tree model')
    model = DecisionTreeClassifier().fit(X_train, y_train)

    # calculate accuracy
    y_hat = model.predict(X_test)
    acc = np.average(y_hat == y_test)
    print('Accuracy:', acc)
    run.log('Accuracy', np.float(acc))

    # calculate AUC
    y_scores = model.predict_proba(X_test)
    auc = roc_auc_score(y_test,y_scores[:,1])
    print('AUC: ' + str(auc))
    run.log('AUC', np.float(auc))

    # Save the trained model
    model_file = 'aviation_model.pkl'
    joblib.dump(value=model, filename=model_file)
    run.upload_file(name = 'outputs/' + model_file, path_or_stream = './' + model_file)

    # Complete the run
    run.complete()

    # Register the model
    if acc > production_accuracy:
        run.register_model(model_path='outputs/aviation_model.pkl', model_name='aviation_model',
                        tags={'Training context':'Inline Training'},
                        properties={'AUC': run.get_metrics()['AUC'], 'Accuracy': run.get_metrics()['Accuracy']})
        print('New model has higher accuracy, hence model trained and registered')
        with open(args.output_new_register_file, "w") as out_file:
                out_file.write('y')
    else:
        print('New model does not have a higher accuracy, hence model trained is not registered')
        with open(args.output_new_register_file, "w") as out_file:
                out_file.write('n')

if __name__ == '__main__':
    main()