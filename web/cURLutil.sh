# common cURL commands for testing REST APIs

# first set the URL you want to send requests to
# export URL="http://localhost:3000/"
export URL="http://panda.d2.comp.nus.edu.sg:3000/"

# GET the root, no need authentication
curl -i -H "Accept: application/json" \
  -H "Content-Type: application/json" \
  -X GET \
  $URL

# GET the token
curl -i "${URL}tokens" \
  -X POST \
  -d 'email=superadmin@rafiki' \
  -d 'password=rafiki'

# the token will expire after a while
# {
#   "token":"eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoiMTA1MTBkZTgtYzZlMi00NmY5LTliOWItZTg2YmU5NzQwYTBjIiwidXNlcl90eXBlIjoiU1VQRVJBRE1JTiIsImV4cCI6MTU4MTUwNzA1Nn0.TeNcGX6fT1s_04SznwNr8D_QXvjwxF3GLPkBxKaClhI",
#   "user_id":"10510de8-c6e2-46f9-9b9b-e86be9740a0c",
#   "user_type":"SUPERADMIN"
# }

export TOKEN="<your token from above>"

#####################################
# Datasets
#####################################

# POST a train dataset from a file
curl -i "${URL}datasets" \
  -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -F name="DS-Train55-$(date +%F_%H-%M-%S)" \
  -F task="IMAGE_CLASSIFICATION" \
  -F dataset=@"/home/svd/Documents/Work/NUS-SOC/FeiyiRafiki//rafiki_panda_dev/OneDrive-2020-02-18/train55.zip"

# POST a test/val dataset from a file
curl -i "${URL}datasets" \
  -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -F name="DS-Val55-$(date +%F_%H-%M-%S)" \
  -F task="IMAGE_CLASSIFICATION" \
  -F dataset=@"/home/svd/Documents/Work/NUS-SOC/FeiyiRafiki/rafiki_panda_dev/OneDrive-2020-02-18/val55.zip"

# GET datasets with authentication (Bearer Token)
# returns an array of objects
curl -i "${URL}datasets" \
  -H "Authorization: Bearer $TOKEN"

#[
# {"datetime_created":"Mon, 10 Feb 2020 11:08:34 GMT","id":"13b48188-a91d-4fa4-9819-f004ba107baf","name":"HI","size_bytes":11338233,"stat":{"feature A":500,"feature B":5000,"feature C":1000},"store_dataset_id":"d60308ce-5f66-4470-98a4-dbaab01ea231.data","task":"IMAGE_CLASSIFICATION"},
# {"datetime_created":"Mon, 10 Feb 2020 11:15:57 GMT","id":"f621abce-ad93-4e52-867b-d9ba27f74890","name":"HI2","size_bytes":11338233,"stat":{"feature A":500,"feature B":5000,"feature C":1000},"store_dataset_id":"ae5b2467-20e8-4aaa-894e-457f203dab81.data","task":"IMAGE_CLASSIFICATION"},
# {"datetime_created":"Tue, 11 Feb 2020 07:00:09 GMT","id":"804a6828-7583-437e-93b0-9e75055350c8","name":"asdfasdfasdfasdfasdf","size_bytes":953,"stat":{"feature A":500,"feature B":5000,"feature C":1000},"store_dataset_id":"883ca65c-b630-49f2-8170-e5b6f830fbc2.data","task":"IMAGE_CLASSIFICATION"}
#]

#####################################
# Models
#####################################

# POST a new model from a file
# no need to include user_id again
curl -i "${URL}models" \
  -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -F name="PyPandaDenseNet-Model-$(date +%F_%H-%M-%S)" \
  -F task="IMAGE_CLASSIFICATION" \
  -F model_file_bytes=@"/home/svd/Documents/Work/NUS-SOC/FeiyiRafiki/rafiki_panda_dev/examples/models/image_classification/PyPandaDenseNet.py" \
  -F model_class="PyPandaDenseNet" \
  -F dependencies='{"torch":"1.0.1","torchvision":"0.2.2","matmatplotlib":"3.1.0","lime":"0.1.1.36"}'

curl -i "${URL}models" \
  -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -F name="PyPandaVgg-Model-$(date +%F_%H-%M-%S)" \
  -F task="IMAGE_CLASSIFICATION" \
  -F model_file_bytes=@"/home/svd/Documents/Work/NUS-SOC/FeiyiRafiki/rafiki_panda_dev/examples/models/image_classification/PyPandaVgg.py" \
  -F model_class="PyPandaVgg" \
  -F dependencies='{"torch":"1.0.1","torchvision":"0.2.2","matmatplotlib":"3.1.0","lime":"0.1.1.36"}'

curl -i "${URL}models" \
  -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -F name='TfFeedForward' \
  -F task='IMAGE_CLASSIFICATION' \
  -F model_file_bytes=@'/home/svd/Documents/Work/NUS-SOC/FeiyiRafiki/rafiki_panda_dev/examples/models/image_classification/TfFeedForward.py' \
  -F model_class='TfFeedForward' \
  -F dependencies='{ "tensorflow": "1.12.0" }'

# GET a single model through model_id
curl -i "${URL}models/2c46d0ca-6b07-4a0e-9f45-a8e3a6f7dc1c" \
  -X GET \
  -H "Authorization: Bearer $TOKEN"

# DELETE a model by model_id
curl -i "${URL}models/0d09ea27-f595-411b-b3ff-dbdab3f0c53c"  \
  -X DELETE \
  -H "Authorization: Bearer $TOKEN"

# GET available models with Bearer Token
# returns an array
curl -i "${URL}models/available" \
  -H "Authorization: Bearer $TOKEN"

# GET available models with specific task
curl -i "${URL}models/available?task=IMAGE_CLASSIFICATION" \
  -H "Authorization: Bearer $TOKEN"

# GET recommended models with Bearer Token
# returns an array
curl -i "${URL}models/recommended?dataset_id=xxx" \
  -H "Authorization: Bearer $TOKEN"

#####################################
# Train Jobs
#####################################

# POST Train Jobs
# budget values MUST NOT be in quotes!
# model_ids should be literal list!
curl -i "${URL}train_jobs" \
  -H "Authorization: Bearer $TOKEN" \
  -F app="TrainJobs-$(date +%F_%H-%M-%S)-app" \
  -F task="IMAGE_CLASSIFICATION" \
  -F train_dataset_id="278927ac-e87d-485b-98d9-eb101f787b2f" \
  -F val_dataset_id="fc72f670-8d25-4bbe-a7e7-dff9e00e3666" \
  -F budget='{ "MODEL_TRIAL_COUNT": 5 }' \
  -F model_ids='["338a6ebd-ecf2-459b-8a6c-86493321eb90"]' \
  -F train_args='{}'

# GET Train JObs
curl -i "${URL}train_jobs" \
  -H "Authorization: Bearer $TOKEN"

# GET Train JObs by app name
curl -i "${URL}train_jobs/TrainJobs-2020-02-22_11-46-46-app" \
  -H "Authorization: Bearer $TOKEN"

# GET Train JObs by app name and app_version
curl -i "${URL}train_jobs/dummyTrainJobs-24182-app/1" \
  -H "Authorization: Bearer $TOKEN"

#####################################
# Trials
#####################################

# GET a trial from train_jobs
curl -i "${URL}train_jobs/TrainJobs-2020-02-22_11-46-46-app/1/trials" \
  -H "Authorization: Bearer $TOKEN"

# in web/src/containers/Jobs/TrialDetails.js
# rafikiClient is calling const data = await this._get(`/trials/${trialId}/logs`)
# GET the "logs" from a trial
curl -i "${URL}trials/{trial_id}/logs" \
  -H "Authorization: Bearer $TOKEN"

# in web/src/containers/Jobs/TrialDetails.js
# rafikiClient is calling const data = await this._get(`/trials/${trialId}`)
# GET trial xxx
curl -i "${URL}trials/{trial_id}" \
  -H "Authorization: Bearer $TOKEN"

#####################################
# Inference Jobs
#####################################

# GET an inference job by appName and appVersion
curl -i "${URL}inference_jobs/PandaApp/1" \
  -H "Authorization: Bearer $TOKEN"

# POST stop an inference job by appName and appVersion
curl -i "${URL}inference_jobs/PandaApp/1/stop" \
  -H "Authorization: Bearer $TOKEN" \
  -X POST

#####################################
# Predictor
#####################################

# the predictor host does not have a fixed port (as of Mar16 2020)
PREDICTOR_PORT='34249'
# and predictor host does not have authentication (as of Mar16 2020)
curl -i http://panda.d2.comp.nus.edu.sg:$PREDICTOR_PORT/predict \
  -X POST \
  -F img=@'/home/svd/Documents/Work/NUS-SOC/FeiyiRafiki/rafiki_panda_dev/examples/data/image_classification/xray_1.png' \

