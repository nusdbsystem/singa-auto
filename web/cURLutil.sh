# common cURL commands for testing REST APIs

# GET the root, no need authentication
curl -i -H "Accept: application/json" \
  -H "Content-Type: application/json" \
  -X GET \
  http://localhost:3000/

# GET the token
curl -i http://localhost:3000/tokens \
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

# POST a new dataset from a file
curl -i http://localhost:3000/datasets \
  -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -F name="dummyDS-$RANDOM" \
  -F task="IMAGE_CLASSIFICATION" \
  -F dataset=@"/home/svd/Downloads/Solutions.zip"

# GET datasets with authentication (Bearer Token)
# returns an array of objects
curl -i http://localhost:3000/datasets \
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
curl -i http://localhost:3000/models \
  -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -F name="dummyModel-$RANDOM" \
  -F task="IMAGE_CLASSIFICATION" \
  -F model_file_bytes=@"/home/svd/Documents/Work/NUS-SOC/FeiyiRafiki/rafiki_panda_dev/examples/models/image_classification/PyPandaVgg.py" \
  -F model_class="PyPandaVgg" \
  -F dependencies='{"torch":"2.0.1","torchvision":"0.2.2"}'

# GET a single model through model_id
curl -i http://localhost:3000/models/2c46d0ca-6b07-4a0e-9f45-a8e3a6f7dc1c \
  -H "Authorization: Bearer $TOKEN"

# GET available models with Bearer Token
# returns an array
curl -i http://localhost:3000/models/available \
  -H "Authorization: Bearer $TOKEN"

# GET available models with specific task
curl -i http://localhost:3000/models/available?task=IMAGE_CLASSIFICATION \
  -H "Authorization: Bearer $TOKEN"

# GET recommended models with Bearer Token
# returns an array
curl -i http://localhost:3000/models/recommended \
  -H "Authorization: Bearer $TOKEN"
  -...

#####################################
# ...
#####################################
