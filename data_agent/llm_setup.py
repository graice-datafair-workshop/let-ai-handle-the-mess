import os

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse

load_dotenv()

token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")

if not token:
    print("AWS_BEARER_TOKEN_BEDROCK not found in environment.")
    print("   Make sure you have a .env file with the token.")
else:
    print("AWS Bearer Token loaded")

bedrock_client = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1",
    config=Config(signature_version=UNSIGNED),
)


def add_bearer_token(request, **kwargs):
    request.headers["Authorization"] = f"Bearer {token}"


bedrock_client.meta.events.register("before-send.bedrock-runtime.*", add_bearer_token)

llm = ChatBedrockConverse(
    model="us.anthropic.claude-sonnet-4-20250514-v1:0",
    client=bedrock_client,
)
