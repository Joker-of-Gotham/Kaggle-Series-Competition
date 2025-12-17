import kaggle_evaluation.core.templates

import default_gateway


class DefaultInferenceServer(kaggle_evaluation.core.templates.InferenceServer):
    def _get_gateway_for_test(self, data_paths=None, file_share_dir=None):
        return default_gateway.DefaultGateway(data_paths)
