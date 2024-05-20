from pathlib import Path
import pandas as pd
import biogeme.database as db
from biogeme.results import bioResults
from simba.mobi.choice.models.mobility_tools.public_transport_subscription_ownership_adults.data_loader import get_data
from simba.mobi.choice.models.mobility_tools.public_transport_subscription_ownership_adults.descriptive_stats import visualize_piecewise_age
from simba.mobi.choice.models.mobility_tools.public_transport_subscription_ownership_adults.model_estimation import estimate_model
from simba.mobi.choice.models.mobility_tools.public_transport_subscription_ownership_adults.model_prediction import predict_model
from sklearn.model_selection import train_test_split
import time

start_time = time.time()

path_to_mobi_zones = Path(
        r"../../Zones/mobi-zones.shp"
)
path_to_mtmc = Path(r"../../MZMV")

# data_directory = Path(Path(__file__).parent.parent.parent.paren
# joinpath("data"))
data_directory = Path(r"simba/mobi/choice/data")

input_directory = data_directory.joinpath("input").joinpath(
        "public_transport_subscription_ownership"
)
df_zp = get_data(input_directory, path_to_mtmc, path_to_mobi_zones)
# Split the data into training and testing sets
train_data, test_data = train_test_split(df_zp, test_size=0.2, random_state=42)
output_directory = data_directory.joinpath("output").joinpath(
        "public_transport_subscription_ownership_validation"
)
output_directory.mkdir(parents=True, exist_ok=True)
estimate_model(train_data, output_directory)

print("Reading the estimated parameters")
results = bioResults(pickleFile=output_directory.joinpath("2015/dcm_indivPT_BUGdist06.pickle"))
betas = results.getBetaValues()

database_test = db.Database("indivPT_test", test_data)
predict_model(database_test, betas, output_directory)

print('Simulated probabilities saved in:', output_directory)

print("total time in s", (time.time() - start_time))