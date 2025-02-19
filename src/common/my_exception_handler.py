from drf_standardized_errors.formatter import ExceptionFormatter
from drf_standardized_errors.types import ErrorResponse
from drf_standardized_errors.handler import ExceptionHandler
from django.core.exceptions import ValidationError as DjangoValidationError
from django.db.utils import IntegrityError as DjangoIntegrityError
from rest_framework.exceptions import ValidationError as DRFValidationError, APIException
from rest_framework.serializers import as_serializer_error
import logging

from common.exceptions import ApplicationError
logger = logging.getLogger('app')


class MyExceptionFormatter(ExceptionFormatter):
    def format_error_response(self, error_response: ErrorResponse):
        error = error_response.errors[0]
        return {
            "type": error_response.type,
            "code": error.code,
            "message": error.detail,
            "extra": error.attr,
        }


class MyExceptionHandler(ExceptionHandler):
    def convert_known_exceptions(self, exc: Exception) -> Exception:
        if isinstance(exc, DjangoValidationError):
            return DRFValidationError(as_serializer_error(exc))
        elif isinstance(exc, DjangoIntegrityError):
            logger.exception(exc)
            return ApplicationError({f'{str(exc)}': 'Database error'})
        else:
            return super().convert_known_exceptions(exc)


class SimpleExceptionFormatter(ExceptionFormatter):
    def format_error_response(self, error_response: ErrorResponse):
        error = error_response.errors[0]
        return error.detail


class MultipleExceptionFormatter(ExceptionFormatter):
    def format_error_response(self, error_response: ErrorResponse):
        results = []
        for error in error_response.errors:
            results.append({
                "code": error.code,
                "message": error.detail,
                "extra": error.attr,
            })

        return {
            "type": error_response.type,
            "errors": results
        }


class DRFFormatterType:
    SIMPLE = 1
    SINGLE_ERROR = 2
    MULTIPLE_ERRORS = 3


map_formatter_classes = {
    DRFFormatterType.SINGLE_ERROR: MyExceptionFormatter,
    DRFFormatterType.SIMPLE: SimpleExceptionFormatter,
    DRFFormatterType.MULTIPLE_ERRORS: MultipleExceptionFormatter
}


class MySimpleExceptionHandler(MyExceptionHandler):
    def get_custom_response(self, formatter_type):
        exc = self.convert_known_exceptions(self.exc)
        if self.should_not_handle(exc):
            return None

        exc = self.convert_unhandled_exceptions(exc)
        exception_formatter_class = map_formatter_classes[formatter_type]
        data = exception_formatter_class(exc, self.context, self.exc).run()
        return data


def get_formatted_error_message(error, formatter_type=DRFFormatterType.SINGLE_ERROR):
    if isinstance(error, APIException):
        return MySimpleExceptionHandler(error, {}).get_custom_response(formatter_type) # noqa
    else:
        if formatter_type == DRFFormatterType.SIMPLE:
            return str(error)
        else:
            return {
                'code': 'invalid',
                'message': str(error)
            }
