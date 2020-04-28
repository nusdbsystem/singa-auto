import traceback
from flask import jsonify, Blueprint, g
from singa_auto.meta_store.meta_store import DuplicateModelNameError

errors_bp = Blueprint('errors', __name__)


@errors_bp.errorhandler(Exception)
def handle_error(error):
    traceback.print_exc()
    if type(error) == DuplicateModelNameError:
        return jsonify({'ErrorMsg': 'DuplicateModelNameError'}), 400

    return traceback.format_exc(), 500
