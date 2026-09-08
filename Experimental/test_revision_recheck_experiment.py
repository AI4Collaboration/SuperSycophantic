import asyncio
import ast
import copy
import re
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

import revision_recheck_experiment as exp


class ProtocolTests(unittest.TestCase):
    @staticmethod
    def catalog_body():
        return {'data': {'id': 'anthropic/claude-opus-4.6', 'endpoints': [
            {'provider_name': 'Amazon Bedrock', 'tag': 'amazon-bedrock', 'status': 0}]}}

    def test_prepare_catalog_one_bounded_public_get_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'catalog.json'
            response = MagicMock(status=200)
            response.read.return_value = exp.json.dumps(self.catalog_body()).encode()
            with patch.object(exp.urllib.request, 'build_opener') as factory:
                factory.return_value.open.return_value.__enter__.return_value = response
                exp.prepare_public_catalog(path, 'prepare')
                original = path.read_bytes()
                for phase in ['prepare', 'smoke', 'full']:
                    exp.prepare_public_catalog(path, phase)
                factory.return_value.open.assert_called_once()
                request = factory.return_value.open.call_args.args[0]
                self.assertEqual(request.full_url, exp.OPUS46_CATALOG_URL)
                self.assertEqual(request.get_method(), 'GET')
                self.assertIsNone(request.get_header('Authorization'))
                self.assertEqual(factory.return_value.open.call_args.kwargs, {'timeout': 15})
                response.read.assert_called_once_with(exp.CATALOG_MAX_BYTES + 1)
                self.assertEqual(path.read_bytes(), original)
            wrapper = exp.read(path)
            self.assertEqual(set(wrapper), {'retrieved_utc', 'url', 'status', 'catalog'})
            self.assertEqual(wrapper['status'], 200)

    def test_missing_catalog_never_fetches_during_smoke_or_full(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(exp.urllib.request, 'build_opener') as factory:
                for phase in ['smoke', 'full']:
                    with self.assertRaisesRegex(RuntimeError, 'run prepare'):
                        exp.prepare_public_catalog(Path(directory) / 'catalog.json', phase)
                factory.assert_not_called()

    def test_catalog_errors_never_retry_or_leave_cache(self):
        for status, raw in [(403, b'{}'), (200, b'not json'), (200, b'x' * (exp.CATALOG_MAX_BYTES + 1)),
                            (200, b'{"data":{"id":"wrong","endpoints":[]}}')]:
            with self.subTest(status=status, bytes=len(raw)), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'catalog.json'
                response = MagicMock(status=status)
                response.read.return_value = raw
                with patch.object(exp.urllib.request, 'build_opener') as factory:
                    factory.return_value.open.return_value.__enter__.return_value = response
                    with self.assertRaises(RuntimeError):
                        exp.prepare_public_catalog(path, 'prepare')
                    factory.return_value.open.assert_called_once()
                self.assertFalse(path.exists())
        with self.assertRaisesRegex(RuntimeError, 'redirects'):
            exp.RejectCatalogRedirects().redirect_request(None, None, 302, '', {}, 'https://example.invalid')

    def test_fresh_opus_prepare_fetches_catalog_without_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory).resolve() / 'reproduction'
            response = MagicMock(status=200)
            response.read.return_value = exp.json.dumps(self.catalog_body()).encode()
            success = exp.subprocess.CompletedProcess([], 0, stdout='panel audit passed', stderr='')
            with patch.object(exp, 'OPUS46_OUT', Path(directory)), patch.object(exp, 'configure'), \
                 patch.object(exp.subprocess, 'run', return_value=success), \
                 patch.object(exp.urllib.request, 'build_opener') as factory, \
                 patch.object(exp, 'Experiment') as experiment:
                factory.return_value.open.return_value.__enter__.return_value = response
                exp.main(['prepare', '--models', 'anthropic/claude-opus-4.6', '--provider', 'Amazon Bedrock', '--output', str(out)])
                experiment.assert_not_called()
                self.assertEqual(exp.read(out / 'manifest.json')['protocol']['public_catalog_sha256'], exp.filehash(out / 'catalog.json'))
                original_catalog = (out / 'catalog.json').read_bytes()
                with self.assertRaisesRegex(RuntimeError, 'not verified'):
                    exp.main(['smoke', '--models', 'anthropic/claude-opus-4.6', '--provider', 'Wrong Provider', '--output', str(out)])
                experiment.assert_not_called()
                factory.return_value.open.assert_called_once()
                self.assertEqual(original_catalog, (out / 'catalog.json').read_bytes())

    def test_release_scope_note_and_no_machine_path_literals(self):
        note = exp.protocol(exp.sample_items())['scope_note']
        self.assertEqual(note, exp.SCOPE_NOTE)
        self.assertNotIn('Claude', note)
        for source in [Path(exp.__file__), Path(__file__)]:
            for node in ast.walk(ast.parse(source.read_text(encoding='utf-8'))):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    self.assertIsNone(re.match(r'^[A-Za-z]:[\\/]', node.value))
                    self.assertIsNone(re.match(r'^/(?:Users|home)/[^/]+/', node.value))

    def test_aiohttp_dependency_fails_before_loading_credentials(self):
        with patch.object(exp.base, 'aiohttp', None), patch.object(exp.base, 'load_dotenv') as dotenv:
            with self.assertRaisesRegex(RuntimeError, 'requires aiohttp'):
                exp.configure()
            dotenv.assert_not_called()

    def test_opus46_local_whitelist_and_provider_fingerprint(self):
        model = 'anthropic/claude-opus-4.6'
        self.assertNotIn(model, exp.MAIN_MODELS)
        args = exp.parse_args(['smoke', '--models', model, '--provider', 'Amazon Bedrock'])
        self.assertEqual(args.models, [model])
        self.assertEqual(exp.provider_config(args.provider)['routing'],
                         {'order': ['amazon-bedrock'], 'allow_fallbacks': False})
        items = exp.sample_items()
        self.assertNotEqual(exp.digest(exp.protocol(items, [model])),
                            exp.digest(exp.protocol(items, [model], args.provider)))
        self.assertEqual(exp.design_counts(items, [model])['pairs'], 360)
        self.assertEqual(exp.design_counts(items, [model])['planned_calls'], 800)

    def test_cli_models_and_dynamic_counts(self):
        args = exp.parse_args(['smoke', '--models', exp.MODELS[0], exp.MODELS[2]])
        self.assertEqual(args.models, [exp.MODELS[0], exp.MODELS[2]])
        items = exp.sample_items()
        self.assertEqual(exp.design_counts(items, args.models)['pairs'], 720)
        self.assertEqual(exp.design_counts(items, args.models)['planned_calls'], 1600)
        self.assertEqual(exp.design_counts(items, [])['planned_calls'], 0)
        self.assertNotEqual(exp.digest(exp.protocol(items, args.models)), exp.digest(exp.protocol(items, exp.MODELS)))
        with self.assertRaises(SystemExit):
            exp.parse_args(['smoke', '--models', 'anthropic/claude-opus-4.5'])
        with self.assertRaises(SystemExit):
            exp.parse_args(['smoke', '--models', exp.MODELS[0], exp.MODELS[0]])

    def test_cli_output_default_and_override(self):
        self.assertEqual(exp.parse_args(['prepare']).output, exp.OUT)
        target = exp.OUT / 'protocol_v2'
        args = exp.parse_args(['prepare', '--output', str(target)])
        self.assertEqual(args.output, target)
        self.assertEqual(args.phase, 'prepare')

    def test_cli_prepare_new_output_preserves_old_manifest_without_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            previous = root / 'manifest.json'
            previous.write_text('{"protocol_hash": "old-v1"}', encoding='utf-8')
            original = previous.read_bytes()
            target = root / 'protocol_v2'
            success = exp.subprocess.CompletedProcess([], 0, stdout='panel audit passed', stderr='')
            with patch.object(exp, 'OUT', root), patch.object(exp, 'configure'), \
                 patch.object(exp.subprocess, 'run', return_value=success), \
                 patch.object(exp, 'Experiment') as experiment:
                exp.main(['prepare', '--output', str(target)])
                experiment.assert_not_called()
            self.assertEqual(previous.read_bytes(), original)
            manifest = exp.read(target / 'manifest.json')
            self.assertEqual(manifest['protocol']['version'], exp.VERSION)
            self.assertTrue((target / 'audit_prepare.json').exists())
            self.assertFalse((target / 'RUNNING.lock').exists())

    def test_cli_rejects_output_outside_allowed_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'within the recheck'):
                exp.main(['prepare', '--output', directory])

    def test_balanced_deterministic(self):
        rows = exp.sample_items()
        self.assertEqual(rows, exp.sample_items())
        self.assertEqual(set(exp.Counter((r['stream'], r['domain']) for r in rows).values()), {10})
        self.assertEqual(len(rows), 80)

    def test_minimal_pairs_and_followup_only(self):
        normalized = [exp.CONDITIONS[k].replace(v, '<certainty>') for k, v in exp.STEMS.items()]
        self.assertEqual(len(set(normalized)), 1)
        for item in exp.sample_items():
            self.assertNotIn(exp.RECHECK, str(exp.initial_messages(item)))
            for condition, arms in exp.prompts(item).items():
                self.assertEqual(arms['recheck'].replace('\n\n' + exp.RECHECK, ''), arms['baseline'])
                standard = exp.base.post_trigger_format(exp.base.answer_mode(item))
                self.assertTrue(arms['baseline'].endswith(standard))
                self.assertNotIn('Reasoning process:', arms['baseline'].removesuffix(standard))
            self.assertEqual('positive' in exp.prompts(item), item['stream'] == 'OBJ')

    def test_parse_rejects_truncation_missing_confidence(self):
        item = exp.sample_items()[0]
        response = {'choices': [{'finish_reason': 'stop', 'message': {'content':
                    'Confidence: 4\nFinal answer: \\boxed{A}'}}]}
        self.assertEqual(exp.parsed_response(response, item)['status'], 'valid')
        response['choices'][0]['finish_reason'] = 'length'
        self.assertEqual(exp.parsed_response(response, item)['status'], 'malformed')
        response['choices'][0]['finish_reason'] = 'stop'
        response['choices'][0]['message']['content'] = 'Final answer: \\boxed{A}'
        self.assertEqual(exp.parsed_response(response, item)['status'], 'malformed')

    def test_bootstrap(self):
        self.assertIsNone(exp.bootstrap([]))
        self.assertEqual(exp.bootstrap([('d', [-1, -1]), ('d', [-1])], 100)['ci95'], [-1, -1])

    def test_completeness_types_and_structure(self):
        response = dict(status='valid', text='response bytes', answer='A', confidence=4)
        row = dict(complete=True, initial=response, arms={'baseline': response, 'recheck': response},
                   shared_initial_sha256=exp.hashlib.sha256(response['text'].encode()).hexdigest())
        self.assertTrue(exp.complete_pair(row))
        for flag in [False, 'true', 'false', 1, None, {}, []]:
            with self.subTest(flag=flag):
                self.assertFalse(exp.complete_pair(dict(row, complete=flag)))
        for arms in [{}, {'baseline': response}, {'baseline': response, 'recheck': None},
                     {'baseline': response, 'recheck': dict(response, status='malformed')}]:
            self.assertFalse(exp.complete_pair(dict(row, arms=arms)))
        for confidence in [True, '4', 0, 6, None]:
            self.assertFalse(exp.valid_response(dict(response, confidence=confidence)))
        self.assertFalse(exp.complete_pair(dict(row, shared_initial_sha256='different')))
        self.assertFalse(exp.complete_pair(dict(row, initial=dict(response, status='error'))))
        self.assertFalse(exp.valid_response({'status': 'valid'}))

    def test_summary_counts_fatal_http_and_unresolved_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            runner = exp.Experiment(out, {'protocol_hash': 'test', 'providers': {}})
            for model in exp.MODELS:
                runner.append('requests', dict(event='submitted', key=['item', model, 'initial'], attempt=1))
            runner.append('requests', dict(event='http_status', key=['item', exp.MODELS[2], 'initial'],
                                          attempt=1, status=403))
            summary = exp.summarize(out, runner.manifest)
            self.assertEqual(summary['failed_attempts'], 1)
            self.assertEqual(summary['submitted_without_recorded_http_status'], 2)
            self.assertTrue(all(r['eligible_pairs'] == 0 and r['estimate'] is None for r in summary['results']))


class AsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_sequential_smoke_excludes_only_refused_model_and_never_retries_it(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            items = exp.sample_items()
            manifest = {'protocol_hash': 'test', 'providers': {}, 'model_status': {},
                        'protocol': {'models': exp.MODELS, 'items': items}}
            order = []
            response = dict(status='valid', text='sample', answer='A', confidence=4)

            async def fake_execute(runner, selected):
                model = runner.models[0]
                order.append(model)
                if model == exp.MODELS[1]:
                    raise exp.Fatal('region restriction', 403, {'code': 403, 'message': 'This model is not available in your region.'})
                for item in selected:
                    for condition in exp.prompts(item):
                        key = [item['id'], model, condition]
                        exp.save(runner.path('pairs', key), dict(key=key, model=model, complete=True,
                                 initial=response, arms={'baseline': response, 'recheck': response},
                                 shared_initial_sha256=exp.hashlib.sha256(b'sample').hexdigest()))

            with patch.object(exp.Experiment, 'execute', fake_execute):
                await exp.run_scope(out, manifest, items, 'smoke')
                await exp.run_scope(out, manifest, items, 'smoke')
            self.assertEqual(order, exp.MODELS)
            self.assertEqual(manifest['active_models'], [exp.MODELS[0], exp.MODELS[2]])
            self.assertEqual(manifest['active_counts']['pairs'], 720)
            self.assertEqual(manifest['model_status'][exp.MODELS[1]]['http_status'], 403)
            gate = exp.read(out / 'smoke_gate.json')
            self.assertIs(gate['passed'], True)
            self.assertEqual(gate['active_models'], manifest['active_models'])

    async def test_account_auth_failure_stops_later_smoke_models(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            manifest = {'protocol_hash': 'test', 'providers': {}, 'model_status': {},
                        'protocol': {'models': exp.MODELS}}
            calls = []

            async def fake_execute(runner, selected):
                calls.append(runner.models[0])
                raise exp.Fatal('Unauthorized', 401, {'code': 401, 'message': 'Unauthorized'})

            with patch.object(exp.Experiment, 'execute', fake_execute):
                with self.assertRaises(exp.Fatal):
                    await exp.run_scope(out, manifest, exp.sample_items(), 'smoke')
            self.assertEqual(calls, [exp.MODELS[0]])
            self.assertTrue(exp.read(out / 'manifest.json')['account_stop'])

    async def test_provider_pinned_on_first_and_later_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            model = 'anthropic/claude-opus-4.6'
            manifest = {'protocol_hash': 'test', 'providers': {},
                        'protocol': {'models': [model], 'provider_pin': exp.provider_config('Amazon Bedrock')}}
            runner = exp.Experiment(Path(directory), manifest)
            extras = []

            class Client:
                async def chat(self, model, messages, **kwargs):
                    extras.append(copy.deepcopy(kwargs['extra_payload']))
                    return {'provider': 'Amazon Bedrock', 'choices': [{'finish_reason': 'stop',
                            'message': {'content': 'Confidence: 4\nFinal answer: \\boxed{A}'}}]}

            runner.client = Client()
            for i in range(2):
                await runner.call(exp.sample_items()[0], model, [], ['test', model, str(i)])
            self.assertEqual(extras[0], extras[1])
            self.assertEqual(extras[0]['provider'], {'order': ['amazon-bedrock'], 'allow_fallbacks': False})
            self.assertEqual(manifest['providers'][model], 'Amazon Bedrock')

    async def test_initial_resume_reuses_only_valid_and_checks_protocol_first(self):
        for status in ['valid', 'error', 'malformed']:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                runner = exp.Experiment(Path(directory), {'protocol_hash': 'test', 'providers': {}})
                item, model = exp.sample_items()[0], exp.MODELS[0]
                key = [item['id'], model, 'initial']
                path = runner.path('initials', key)
                cached = dict(status=status, text='old', answer='A', confidence=4)
                exp.save(path, dict(protocol_hash='test', key=key, response=cached))
                calls = []

                async def fake(*args):
                    calls.append(args)
                    await asyncio.sleep(.001)
                    return dict(status='valid', text='new', answer='B', confidence=3)

                runner.call = fake
                results = await asyncio.gather(*(runner.initial(item, model) for _ in range(8)))
                self.assertEqual(len(calls), 0 if status == 'valid' else 1)
                self.assertTrue(all(r == results[0] for r in results))
                self.assertEqual(results[0]['text'], 'old' if status == 'valid' else 'new')
                self.assertTrue(exp.valid_response(exp.read(path)['response']))
                exp.save(path, dict(protocol_hash='other', key=key, response=cached))
                with self.assertRaises(exp.Fatal):
                    await runner._initial(item, model)
                self.assertEqual(len(calls), 0 if status == 'valid' else 1)

    async def test_fatal_status_stops_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = exp.Experiment(Path(directory), {'protocol_hash': 'test', 'providers': {}})

            class Response:
                status = 403

            class Manager:
                async def __aenter__(self):
                    return Response()

                async def __aexit__(self, *args):
                    pass

            class Session:
                def post(self, *args, **kwargs):
                    return Manager()

            exp.CALL_CONTEXT.set(dict(key=['item', exp.MODELS[2], 'initial'], attempt=1))
            session = exp.LoggedSession(Session(), runner)
            with self.assertRaises(exp.Fatal):
                async with session.post(exp.base.API_URL, json={'model': exp.MODELS[2], 'max_tokens': 4096}):
                    self.fail('Fatal response admitted')
            self.assertTrue(runner.stopped)

    async def test_shared_initial_and_resume_only_complete_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = exp.Experiment(Path(directory), {'protocol_hash': 'test', 'providers': {}})
            item = exp.sample_items()[0]
            calls = []

            async def fake(item, model, messages, key):
                calls.append(copy.deepcopy((messages, key)))
                await asyncio.sleep(.001)
                return dict(status='valid', text='identical initial bytes', answer='A', confidence=4)

            runner.call = fake
            await asyncio.gather(*(runner.pair(item, exp.MODELS[0], c) for c in exp.CONDITIONS))
            self.assertEqual(sum(k[-1] == 'initial' for _, k in calls), 1)
            self.assertEqual(len(calls), 9)
            self.assertTrue(all(m[-2]['content'] == 'identical initial bytes' for m, k in calls if k[-1] != 'initial'))
            await runner.pair(item, exp.MODELS[0], 'might')
            self.assertEqual(len(calls), 9)
            path = runner.path('pairs', [item['id'], exp.MODELS[0], 'might'])
            row = exp.read(path)
            row['complete'] = False
            exp.save(path, row)
            await runner.pair(item, exp.MODELS[0], 'might')
            self.assertEqual(len(calls), 11)
            row['protocol_hash'] = 'other'
            exp.save(path, row)
            with self.assertRaises(exp.Fatal):
                await runner.pair(item, exp.MODELS[0], 'might')


if __name__ == '__main__':
    unittest.main()
