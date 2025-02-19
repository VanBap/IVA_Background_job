import logging
from rest_framework.exceptions import APIException, NotFound, ValidationError

logger = logging.getLogger('app')


class ApplicationError(APIException):
    status_code = 500
    default_code = 'application_error'
    default_detail = 'Something went wrong on server'


class ApplicationErrorExtra(APIException):
    default_code = 'application_error'

    def __init__(self, extra_message):
        super().__init__({extra_message: 'Something went wrong on server'})


class NotFoundError(NotFound):
    pass


class ServiceUnavailableError(ApplicationError):
    status_code = 503
    default_code = 'service_unavailable'
    default_detail = 'An internal services was unavailable'


class InvalidInputError(ValidationError):
    default_code = 'invalid'
    default_detail = 'Invalid input params'


class GroupNameAlreadyExist(ValidationError):
    default_code = 'group_name_exist'
    default_detail = 'Name of group already exists'


class ImageInvalid(ValidationError):
    default_code = 'image_invalid'
    default_detail = 'Data is not a valid image'


class MediaFileInvalid(ValidationError):
    default_code = 'media_file_invalid'

    def __init__(self, extra_message):
        super().__init__({extra_message: 'Media file is invalid'})


def wrap_drf_exceptions(func):
    def inner(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except APIException as ex:
            raise ex
        except Exception as ex:
            raise ApplicationError({str(ex): 'Application error'}) from ex

    return inner
