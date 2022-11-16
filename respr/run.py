from respr.pipeline.base import DEFAULT_CONFIG_PATH, REGISTERED_PIPELINES
from loguru import logger

def main():
    import yaml
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument("-c", "--config", default=None, type=str)
    DEFAULT_PIPELINE = "Pipeline2"
    args = ap.parse_args()
    if args.config is None:
        config_path = DEFAULT_CONFIG_PATH
    else:
        config_path = args.config
    config_data = None
    with open(config_path, 'r', encoding="utf-8") as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)
    pipeline_name = config_data["pipeline"]["name"]
    logger.info(f"Using pipeline: {pipeline_name}")
    if pipeline_name is None:
        pipeline_name = DEFAULT_PIPELINE
    pipeline_class = REGISTERED_PIPELINES[pipeline_name]
    p = pipeline_class(config_data)
    p.run()
    
if __name__ == "__main__":
    main()

