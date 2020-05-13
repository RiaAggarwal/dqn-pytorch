import turtle
import numpy as np

## ENVIRONMENT
wn = turtle.Screen()
wn.title('Pong by Yuance')
wn.bgcolor('blue')
wn.setup(width=800, height=600)
wn.tracer(0)

# Score
score_a = 0
score_b = 0

# media red
height = 26
width = 10
media = turtle.Turtle()
media.speed(0)
media.shape('square')
media.color('red')
media.shapesize(stretch_wid=height, stretch_len=width)
media.penup()
media.goto(0,0)

# Paddle A
paddle_a = turtle.Turtle()
paddle_a.speed(0)
paddle_a.shape('square')
paddle_a.color('yellow')
paddle_a.shapesize(stretch_wid=5, stretch_len=1)
paddle_a.penup()
paddle_a.goto(-350,0)

# Paddle B
paddle_b = turtle.Turtle()
paddle_b.speed(0)
paddle_b.shape('square')
paddle_b.color('yellow')
paddle_b.shapesize(stretch_wid=5, stretch_len=1)
paddle_b.penup()
paddle_b.goto(350,0)

# Ball
dx = 0.8
dy = 1
ball = turtle.Turtle()
ball.speed(0)
ball.shape('circle')
ball.color('white')
ball.penup()
ball.goto(0,0)
ball.dx = dx
ball.dy = dy

# words
pen1 = turtle.Turtle()
pen1.speed(0)
pen1.color('white')
pen1.penup()
pen1.hideturtle()
pen1.goto(0, 260)
pen1.write('Player A: {}  Player B: {}'.format(score_a, score_b), align='center', font=('Courier',24,'normal'))

pen2 = turtle.Turtle()
pen2.speed(0)
pen2.color('white')
pen2.penup()
pen2.hideturtle()
pen2.goto(0, -300)
pen2.write('left:{} ball:({:.2f},{:.2f}) right:{}'.format(paddle_a.ycor(), round(ball.xcor()), ball.ycor(), paddle_b.ycor()), align='center', font=('Courier',24,'normal'))

# border
border1 = turtle.Turtle()
border1.speed(0)
border1.shape('square')
border1.color('black')
border1.shapesize(stretch_wid=0.2, stretch_len=50)
border1.penup()
border1.goto(0, 260)

border2 = turtle.Turtle()
border2.speed(0)
border2.shape('square')
border2.color('black')
border2.shapesize(stretch_wid=0.2, stretch_len=50)
border2.penup()
border2.goto(0, -260)

# Function
def paddle_a_up():
    y = paddle_a.ycor()
    if y < 240:
        y += 5
    paddle_a.sety(y)
    
def paddle_a_down():
    y = paddle_a.ycor()
    if y > -240:
        y -= 5
    paddle_a.sety(y)
    
def paddle_b_up():
    y = paddle_b.ycor()
    if y < 240:
        y += 5
    paddle_b.sety(y)
    
def paddle_b_down():
    y = paddle_b.ycor()
    if y > -240:
        y -= 5
    paddle_b.sety(y)

# Main game loop
while True:
    wn.update()
    # Move ball
    ball.setx(ball.xcor() + ball.dx)
    ball.sety(ball.ycor() + ball.dy)

    # move computer paddle
    if np.random.rand() < 0.2:
        if paddle_a.ycor() < ball.ycor():
            paddle_a_up()
        else:
            paddle_a_down()
'''
    if paddle_b.ycor() < ball.ycor() + np.random.normal(0,150):
        paddle_b_up()
    else:
        paddle_b_down()
'''
    if np.random.rand() < 0.1:
        pen2.clear()
        pen2.write('left:{} ball:({:.2f},{:.2f}) right:{}'.format(paddle_a.ycor(), ball.xcor(), ball.ycor(), paddle_b.ycor()), align='center', font=('Courier',24,'normal'))

    # Border check
    if ball.ycor() > 250:
        ball.sety(250)
        ball.dy *= -1
        
    if ball.ycor() < -250:
        ball.sety(-250)
        ball.dy *= -1
        
    if ball.xcor() > 390:
        ball.goto(0,0)
        ball.dx = - dx * np.sign(ball.dx)
        ball.dy = dy * np.sign(ball.dy)
        score_a += 1
        pen1.clear()
        pen1.write('paddle a: {}  paddle b: {}'.format(score_a, score_b), align='center', font=('Courier',24,'normal'))
        
    if ball.xcor() < -390:
        ball.goto(0,0)
        ball.dx = dx
        ball.dy = dy
        score_b += 1
        pen1.clear()
        pen1.write('paddle a: {}  paddle b: {}'.format(score_a, score_b), align='center', font=('Courier',24,'normal'))
        
    # Collisions
    if 350 > ball.xcor() > 340 and paddle_b.ycor() - 45 < ball.ycor() < paddle_b.ycor() + 45:
        ball.setx(340)
        ball.dx *= -1
        
    if -340 > ball.xcor() > -350 and paddle_a.ycor() - 45 < ball.ycor() < paddle_a.ycor() + 45:
        ball.setx(-340)
        ball.dx *= -1

    # deflection
    if -100.2 < ball.xcor() < -99.8 and np.sign(ball.dx) == 1:
        ball.dx *= 0.5
        ball.dy *= 0.8
    if 99.8 < ball.xcor() < 100.2 and np.sign(ball.dx) == 1:
        ball.dx *= 2
        ball.dy *= 1.25
    if -100.2 < ball.xcor() < -99.8 and np.sign(ball.dx) == -1:
        ball.dx *= 2
        ball.dy *= 1.25
    if 99.8 < ball.xcor() < 100.2 and np.sign(ball.dx) == -1:
        ball.dx *= 0.5
        ball.dy *= 0.8

