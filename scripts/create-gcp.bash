#!/bin/bash
set -exu

export MYINSNAME="my-gpu-cuda-test-instance-abc-1"
export PROJNAME="testbed-358013"

docker pull gcr.io/google.com/cloudsdktool/google-cloud-cli:latest

# first time only. preserves the volume
docker run -ti --name gcloud-config gcr.io/google.com/cloudsdktool/google-cloud-cli gcloud auth login \
	|| echo "skipped login"


docker run --rm --volumes-from gcloud-config gcr.io/google.com/cloudsdktool/google-cloud-cli gcloud compute instances list --project $PROJNAME
docker run --rm --volumes-from gcloud-config gcr.io/google.com/cloudsdktool/google-cloud-cli gcloud config set project $PROJNAME

echo "skip creation" || \
docker run -it   --rm --volumes-from gcloud-config  gcr.io/google.com/cloudsdktool/google-cloud-cli:latest \
  gcloud compute instances create $MYINSNAME \
    --project=$PROJNAME \
    --zone=us-central1-c --machine-type=n1-standard-1 --network-interface=network-tier=PREMIUM,subnet=default \
    --maintenance-policy=TERMINATE --provisioning-model=STANDARD --service-account=80405641773-compute@developer.gserviceaccount.com \
    --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append \
    --accelerator=count=1,type=nvidia-tesla-t4 --tags=http-server,https-server \
    --create-disk=auto-delete=yes,boot=yes,device-name=$MYINSNAME,image=projects/debian-cloud/global/images/debian-11-bullseye-v20220719,mode=rw,size=10,type=projects/$PROJNAME/zones/us-central1-a/diskTypes/pd-balanced \
    --no-shielded-secure-boot --shielded-vtpm --shielded-integrity-monitoring --reservation-affinity=any


