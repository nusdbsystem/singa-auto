from flask import jsonify, Blueprint, g
from rafiki.utils.auth import auth
from rafiki.utils.requests_params import param_check

events_bp = Blueprint('events', __name__)


@events_bp.route('/actions/stop_all_jobs', methods=['POST'])
@auth([])
def stop_all_jobs(auth):
    admin = g.admin

    with admin:
        train_jobs = admin.stop_all_train_jobs()
        inference_jobs = admin.stop_all_inference_jobs()
        return jsonify({
            'train_jobs': train_jobs,
            'inference_jobs': inference_jobs
        })

####################################
# Internal Events
####################################

@events_bp.route('/event/<name>', methods=['POST'])
@auth([])
@param_check(required_parameters={})
def handle_event(auth, params, name,):
    admin = g.admin
    with admin:
        return jsonify(admin.handle_event(name, **params))



