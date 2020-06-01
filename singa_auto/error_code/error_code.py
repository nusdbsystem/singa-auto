import json
import traceback

class SingaAutoBaseResponse(object):
    def __init__(self, success=0, error_code=0, message='Result is Success', data=None):
        # super(SingaAutoBaseResponse, self).__init__()
        self.error_code = error_code
        self.success = success
        self.message = message
        self.data = data

    def __str__(self):
        return json.dumps(self.__dict__)

    def __repr__(self):
        return '{}:{}'.format(self.__class__.__name__,  json.dumps(self.__dict__))

    def __iter__(self):
        for item in self.__dict__:
            yield (item, self.__dict__[item])

class SingaAutoBaseException(SingaAutoBaseResponse, Exception):
    def __init__(self, success=1, error_code=500, message='Base Error', data=None):
        super(SingaAutoBaseException, self).__init__(success=success, \
            error_code=error_code, message=message, data=data)

class ResultSuccess(SingaAutoBaseResponse):
    def __init__(self, data=None):
        super(ResultSuccess, self).__init__(data=data)

class SystemInternalError(SingaAutoBaseException):
    def __init__(self, message='System Internal Error'):
        super(SystemInternalError, self).__init__(error_code=500, \
            message=message, data=traceback.format_exc())

class InvalidAuthorizationHeaderError(SingaAutoBaseException):
    def __init__(self, message="Invalid authorization header."):
        super(InvalidAuthorizationHeaderError, self).__init__(error_code=8000, \
            message=message)

class InvalidParamsError(SingaAutoBaseException):
    def __init__(self, message="Invalid params."):
        super(InvalidParamsError, self).__init__(error_code=8001, message=message)

class InvalidParamsFormatError(SingaAutoBaseException):
    def __init__(self, message="Invalid params format."):
        super(InvalidParamsFormatError, self).__init__(error_code=8002, message=message)

class InvalidPasswordError(SingaAutoBaseException):
    def __init__(self, message="Invalid password."):
        super(InvalidPasswordError, self).__init__(error_code=8003, message=message)

class InvalidQueryFormatError(SingaAutoBaseException):
    def __init__(self, message="Invalid query format."):
        super(InvalidQueryFormatError, self).__init__(error_code=8004, message=message)

class InvalidUserError(SingaAutoBaseException):
    def __init__(self, message="Invalid user."):
        super(InvalidUserError, self).__init__(error_code=8005, message=message)

class InvalidUserTypeError(SingaAutoBaseException):
    def __init__(self, message="Invalid user type."):
        super(InvalidUserTypeError, self).__init__(error_code=8006, message=message)

class UnauthorizedError(SingaAutoBaseException):
    def __init__(self, message="Unauthorized."):
        super(UnauthorizedError, self).__init__(error_code=8007, message=message)

class UserAlreadyBannedError(SingaAutoBaseException):
    def __init__(self, message="User already banned."):
        super(UserAlreadyBannedError, self).__init__(error_code=8008, message=message)

class UserExistsError(SingaAutoBaseException):
    def __init__(self, message="User already exists."):
        super(UserExistsError, self).__init__(error_code=8009, message=message)

class InvalidDatasetError(SingaAutoBaseException):
    def __init__(self, message="Invalid dataset."):
        super(InvalidDatasetError, self).__init__(error_code=8010, message=message)

class DuplicateModelNameError(SingaAutoBaseException):
    def __init__(self, message="Duplicated model name."):
        super(DuplicateModelNameError, self).__init__(error_code=8011, message=message)

class InvalidDAGError(SingaAutoBaseException):
    def __init__(self, message="Invalid DAG."):
        super(InvalidDAGError, self).__init__(error_code=8012, message=message)

class InvalidInferenceJobError(SingaAutoBaseException):
    def __init__(self, message="Invalid inference job."):
        super(InvalidInferenceJobError, self).__init__(error_code=8013, message=message)

class InvalidModelAccessRightError(SingaAutoBaseException):
    def __init__(self, message="Invalid model access right."):
        super(InvalidModelAccessRightError, self).__init__(error_code=8014, message=message)

class InvalidModelClassError(SingaAutoBaseException):
    def __init__(self, message="Invalid model class."):
        super(InvalidModelClassError, self).__init__(error_code=8015, message=message)

class InvalidModelError(SingaAutoBaseException):
    def __init__(self, message="Invalid model."):
        super(InvalidModelError, self).__init__(error_code=8016, message=message)

class InvalidRunningInferenceJobError(SingaAutoBaseException):
    def __init__(self, message="Invalid running inference job."):
        super(InvalidRunningInferenceJobError, self).__init__(error_code=8017, message=message)

class InvalidTrainJobError(SingaAutoBaseException):
    def __init__(self, message="Invalid train job."):
        super(InvalidTrainJobError, self).__init__(error_code=8018, message=message)

class InvalidTrialError(SingaAutoBaseException):
    def __init__(self, message="Invalid trial."):
        super(InvalidTrialError, self).__init__(error_code=8019, message=message)

class InvalidSubTrainJobError(SingaAutoBaseException):
    def __init__(self, message="Invalid sub train job"):
        super(InvalidSubTrainJobError, self).__init__(error_code=8020, message=message)

class InvalidWorkerError(SingaAutoBaseException):
    def __init__(self, message="Invalid worker"):
        super(InvalidWorkerError, self).__init__(error_code=8021, message=message)

class NoModelsForTrainJobError(SingaAutoBaseException):
    def __init__(self, message="No model for train job."):
        super(NoModelsForTrainJobError, self).__init__(error_code=8022, message=message)

class ModelUsedError(SingaAutoBaseException):
    def __init__(self, message="Model is being used."):
        super(ModelUsedError, self).__init__(error_code=8023, message=message)

class RunningInferenceJobExistsError(SingaAutoBaseException):
    def __init__(self, message="Running inference job already exists."):
        super(RunningInferenceJobExistsError, self).__init__(error_code=8024, message=message)

class UnsupportedKnobConfigError(SingaAutoBaseException):
    def __init__(self, message="Unsupported knob config."):
        super(UnsupportedKnobConfigError, self).__init__(error_code=8025, message=message)

class UnsupportedKnobError(SingaAutoBaseException):
    def __init__(self, message="Unsupported knob."):
        super(UnsupportedKnobError, self).__init__(error_code=8026, message=message)

class InvalidServiceRequestError(SingaAutoBaseException):
    def __init__(self, message="Invalid service request."):
        super(InvalidServiceRequestError, self).__init__(error_code=8027, message=message)

class SingaAutoConnectionError(SingaAutoBaseException):
    def __init__(self, message="Singa-auto connection error."):
        super(SingaAutoConnectionError, self).__init__(error_code=8028, message=message)

class ServiceDeploymentError(SingaAutoBaseException):
    def __init__(self, message="Service deployment error."):
        super(ServiceDeploymentError, self).__init__(error_code=8029, message=message)

class LoginExpireError(SingaAutoBaseException):
    def __init__(self, message="Login Expire error."):
        super(LoginExpireError, self).__init__(error_code=8030, message=message)

class TokenError(SingaAutoBaseException):
    def __init__(self, message="Token error."):
        super(TokenError, self).__init__(error_code=8031, message=message) 

class ServiceRequestError(SingaAutoBaseException):
    def __init__(self, message="Service request error."):
        super(ServiceRequestError, self).__init__(error_code=8032, message=message) 

class InvalidDatasetFormatException(SingaAutoBaseException):
    def __init__(self, message="Invalid dataset format error."):
        super(InvalidDatasetFormatException, self).__init__(error_code=8033, message=message) 

class InvalidDAGException(SingaAutoBaseException):
    def __init__(self, message="Invalid DAG error."):
        super(InvalidDAGException, self).__init__(error_code=8034, message=message) 

mapError = {
    0: ResultSuccess,
    500: SystemInternalError,
    8000: InvalidAuthorizationHeaderError,
    8001: InvalidParamsError,
    8002: InvalidParamsFormatError,
    8003: InvalidPasswordError,
    8004: InvalidQueryFormatError,
    8005: InvalidUserError,
    8006: InvalidUserTypeError,
    8007: UnauthorizedError,
    8008: UserAlreadyBannedError,
    8009: UserExistsError,
    8010: InvalidDatasetError,
    8011: DuplicateModelNameError,
    8012: InvalidDAGError,
    8013: InvalidInferenceJobError,
    8014: InvalidModelAccessRightError,
    8015: InvalidModelClassError,
    8016: InvalidModelError,
    8017: InvalidRunningInferenceJobError,
    8018: InvalidTrainJobError,
    8019: InvalidTrialError,
    8020: InvalidSubTrainJobError,
    8021: InvalidWorkerError,
    8022: NoModelsForTrainJobError,
    8023: ModelUsedError,
    8024: RunningInferenceJobExistsError,
    8025: UnsupportedKnobConfigError,
    8026: UnsupportedKnobError,
    8027: InvalidServiceRequestError,
    8028: SingaAutoConnectionError,
    8029: ServiceDeploymentError,
    8030: LoginExpireError,
    8031: TokenError,
    8032: ServiceRequestError,
    8033: InvalidDatasetFormatException,
    8034: InvalidDAGException
}

def generate_error(error_code):
    return mapError[error_code]()
