# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 07:57:18 2016

@author: Tom
"""
import click


@click.command()
@click.option('--count', default=1, help='number of greetings')
@click.argument('name')
def hello(count, name):
    for x in range(count):
        click.echo('Hello %s!' % name)


if __name__ == '__main__':
    hello()
