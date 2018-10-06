from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import argparse
import model.config as config
from gerbil.nn_processing import NNProcessing
from model.util import load_train_args
from gerbil.build_entity_universe import BuildEntityUniverse


class GetHandler(BaseHTTPRequestHandler):

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        self.send_response(200)
        self.end_headers()

        if args.build_entity_universe:
            buildEntityUniverse.process(*read_json(post_data))
            response = []
        else:
            response = nnprocessing.process(*read_json(post_data))

        print("response in server.py code:\n", response)
        self.wfile.write(bytes(json.dumps(response), "utf-8"))
        return


def read_json(post_data):
    data = json.loads(post_data.decode("utf-8"))
    #print("received data:", data)
    text = data["text"]
    spans = [(int(j["start"]), int(j["length"])) for j in data["spans"]]
    return text, spans


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", default="per_document_no_wikidump",
                        help="under folder data/tfrecords/")
    parser.add_argument("--training_name", default="doc_fixed_nowiki_evecsl2dropout")
    parser.add_argument("--all_spans_training", type=bool, default=False)
    parser.add_argument("--el_mode", dest='el_mode', action='store_true')
    parser.add_argument("--ed_mode", dest='el_mode', action='store_false')
    parser.set_defaults(el_mode=True)

    parser.add_argument("--running_mode", default=None, help="el_mode or ed_mode, so"
                                "we can restore an ed_mode model and run it for el")

    parser.add_argument("--lowercase_spans_pem", type=bool, default=False)

    parser.add_argument("--entity_extension", default=None, help="extension_entities or extension_entities_all etc")

    # those are for building the entity set
    parser.add_argument("--build_entity_universe", type=bool, default=False)
    parser.add_argument("--hardcoded_thr", type=float, default=None, help="0, 0.2")
    parser.add_argument("--el_with_stanfordner_and_our_ed", type=bool, default=False)

    parser.add_argument("--persons_coreference", type=bool, default=False)
    parser.add_argument("--persons_coreference_merge", type=bool, default=False)

    args = parser.parse_args()
    if args.persons_coreference_merge:
        args.persons_coreference = True
    print(args)
    if args.build_entity_universe:
        return args, None

    temp = "all_spans_" if args.all_spans_training else ""
    args.experiment_folder = config.base_folder+"data/tfrecords/" + args.experiment_name+"/"

    args.output_folder = config.base_folder+"data/tfrecords/" + \
                         args.experiment_name+"/{}training_folder/".format(temp) + \
                         args.training_name+"/"

    train_args = load_train_args(args.output_folder, "gerbil")
    train_args.entity_extension = args.entity_extension

    print(train_args)
    return args, train_args


def terminate():
    tee.close()
    if args.build_entity_universe:
        buildEntityUniverse.flush_entity_universe()
    else:
        print("from_myspans_to_given_spans_map_errors:", nnprocessing.from_myspans_to_given_spans_map_errors)


if __name__ == "__main__":
    args, train_args = _parse_args()
    if args.build_entity_universe:
        buildEntityUniverse = BuildEntityUniverse()
    else:
        nnprocessing = NNProcessing(train_args, args)
    server = HTTPServer(('localhost', 5555), GetHandler)
    print('Starting server at http://localhost:5555')
    from model.util import Tee
    tee = Tee('server.txt', 'w')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        terminate()
        exit(0)
