docker-compose up -d
sleep 10

export S3_ENDPOINT_URL='http://localhost:4566'
export INPUT_FILE_PATTERN='s3://nyc-duration/in/{year:04d}-{month:02d}.parquet'
export OUTPUT_FILE_PATTERN='s3://nyc-duration/out/{year:04d}-{month:02d}.parquet'

aws --endpoint-url="${S3_ENDPOINT_URL}" s3 mb s3://nyc-duration

pipenv run python integration_test_q6.py

ERROR_CODE=$?

if [ ${ERROR_CODE} != 0 ]; then
    docker-compose logs
    docker-compose down
    exit ${ERROR_CODE}
fi

echo "Integration test completed!"

docker-compose down