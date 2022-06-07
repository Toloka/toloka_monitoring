import os
import datetime

import toloka.client as toloka
import toloka.client.project.template_builder as tb
from toloka.client.primitives.operators import CompareOperator
from toloka_monitoring.config import TOLOKA_API_TOKEN


def create_project():
    project = toloka.Project(
        public_name='Cat or dog?',
        public_description='Is this the image of a dog or a cat?',
        private_comment='Cat vs Dog monitoring'
    )

    input_specification = {'image_url': toloka.project.UrlSpec(), 'pred_id': toloka.project.StringSpec()}
    output_specification = {'label': toloka.project.StringSpec()}

    image_viewer = tb.ImageViewV1(tb.InputData('image_url'),
                                  ratio=[1, 1],
                                  popup=False,
                                 )

    label_buttons = [
        tb.GroupFieldOption('cat', 'Cat'),
        tb.GroupFieldOption('dog', 'Dog'),
        tb.GroupFieldOption('neither', 'Neither'),
    ]

    radio_group_field = tb.ButtonRadioGroupFieldV1(
        tb.OutputData('label'),
        label_buttons,
        validation=tb.RequiredConditionV1(),
    )

    task_width_plugin = tb.TolokaPluginV1(
        'scroll',
        task_width=300,
    )

    hot_keys_plugin = tb.HotkeysPluginV1(
        key_1=tb.SetActionV1(tb.OutputData('label'), 'cat'),
        key_2=tb.SetActionV1(tb.OutputData('label'), 'dog'),
        key_3=tb.SetActionV1(tb.OutputData('label'), 'neither'),
    )

    project_interface = toloka.project.TemplateBuilderViewSpec(
        config=tb.TemplateBuilder(
            view=tb.ListViewV1([image_viewer, radio_group_field]),
            plugins=[task_width_plugin, hot_keys_plugin],
        )
    )

    project.task_spec = toloka.project.task_spec.TaskSpec(
        input_spec=input_specification,
        output_spec=output_specification,
        view_spec=project_interface,
    )

    project.public_instructions = """
    In this task, you will see images that might contain one of:<br/>
    1. Cats,
    2. Dogs,
    3. Something else.
    Your task is to classify these images.<br/>

    <b>Some images are blurry and hard to label</b>. That's the nature of the task, so just assign whatever label seems most appropriate.

    How to complete the task:
    <ul>
    <li>Look at the picture.</li>
    <li>Click on the image to resize it. You can rotate the image if it's in the wrong orientation.</li>
    <li>Chose one of the possible answers. If the picture is unavailable or you have any other technical difficulty, please write us about it.</li>
    <li>If you think that you can not classify the image correctly, choose the something else option.</li>
    <li>You can use keyboard shortcuts (1 for cat, 2 for dog, 3 for something else) to pick labels.</li>
    </ul>
    """.strip()

    return project


def create_pool(toloka_client, project_id):
    global_skill = toloka_client.get_skill("25627")
    pool = toloka.Pool(
        project_id=project_id,
        private_name='Monitoring pool',
        may_contain_adult_content=False,
        reward_per_assignment=0.01,
        assignment_max_duration_seconds=60 * 5,
        will_expire=datetime.datetime.utcnow() + datetime.timedelta(days=365),
    )
    pool.defaults = toloka.pool.Pool.Defaults(
        default_overlap_for_new_tasks=3,
        default_overlap_for_new_task_suites=3,
    )
    pool.set_mixer_config(
        real_tasks_count=10,
    )
    pool.filter = toloka.filter.Languages.in_('EN') & toloka.filter.FilterOr([
        toloka.filter.Skill(key=global_skill.id,
                            operator=CompareOperator.GT,
                            value=30),
    ])
    pool = toloka_client.create_pool(pool)
    return pool


if __name__ == '__main__':
    toloka_client = toloka.TolokaClient(TOLOKA_API_TOKEN, 'PRODUCTION')

    project = create_project()
    project = toloka_client.create_project(project)

    print(f'Toloka project: {project.id}')
