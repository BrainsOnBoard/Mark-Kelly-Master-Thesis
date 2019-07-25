import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mathutils import Vector, Matrix
from tqdm import tqdm

np.set_printoptions(linewidth=1000)
plt.style.use('ggplot')


class Predator:
    register = []
    done = 0

    def __init__(self,
                 target = None,
                 initial_position = Vector((0,0)),
                 initial_phi = 0,
                 initial_psi = np.pi/2,
                 speed = 0.1,
                 pursuit_mode = 'direct',
                 alpha = 0.05,
                 beta = 0.05,
                 wind_speed = 0.0,
                 wind_direction = Vector((1,-1)),
                 colour = 'r',
                 intercepted = False
                 ):

        self.target = target
        self.position = initial_position
        self.phi = initial_phi
        self.prev_phi = initial_phi
        self.psi = initial_psi
        self.speed = speed
        self.velocity = self.speed * world.simulation.dt *Vector((np.cos(self.psi), np.sin(self.psi)))
        self.heading = Vector((np.cos(self.psi + self.phi), np.sin(self.psi + self.phi)))
        self.intercepted = intercepted

        self.pursuit_mode = pursuit_mode
        self.alpha = alpha
        self.beta = beta

        self.wind = wind_speed * world.simulation.dt *wind_direction.normalized()

        self.colour = colour
        self.x_positions = [self.position.x]
        self.y_positions = [self.position.y]

        if world.simulation.animate is True:
            self.path, = plt.plot(self.x_positions, self.y_positions, lw=0.5)#, c=self.colour)
            if self.pursuit_mode == 'direct':
                self.plot = plt.scatter(self.position.x, self.position.y, marker='s')#, c=self.colour)
            else:
                self.plot = plt.scatter(self.position.x, self.position.y, marker='^')  # , c=self.colour)

        Predator.register.append(self)


    def bearing(self):
        target_position = self.target.position - self.position
        # if target_position.magnitude < 10:
        #     Prey.register[0].speed = 0.4
        if target_position.magnitude < 0.5:
            self.intercepted = True
            Predator.done += 1
            world.simulation.times[self.pursuit_mode][np.where(world.simulation.betas[self.pursuit_mode] == self.beta)[0][0]] \
                                               [np.where(world.simulation.alphas == self.alpha)[0][0]] \
                                                    = world.simulation.t

        return target_position.angle_signed(self.heading)

    def update_phi(self):
        self.prev_phi = self.phi
        self.phi += self.alpha * self.bearing()
        self.heading = Vector((np.cos(self.phi + self.psi), np.sin(self.phi + self.psi)))

        self.phi = (self.phi + np.pi) % (2 * np.pi) - np.pi  # wraps angles around so -1.5 pi = 0.5 pi

    def update_psi(self):
        #  Check if head is rotated by maximum amount and update body to keep head at maximum
        max_angle = np.pi
        if self.phi > max_angle:
            self.psi += self.phi - max_angle
            self.phi = max_angle
        elif self.phi < - max_angle:
            self.psi += self.phi + max_angle
            self.phi = - max_angle
        else:  #
            if self.pursuit_mode == 'direct':
                self.psi += self.beta * self.phi
                self.phi -= self.beta * self.phi
            elif self.pursuit_mode == 'bearing':
                self.psi += self.beta * (self.phi - self.prev_phi)
                self.phi -= self.beta * (self.phi - self.prev_phi)
            else:
                raise TypeError('No Pursuit Mode Specified')

        self.psi = (self.psi + np.pi) % (2 * np.pi) - np.pi

    def update_velocity(self):
        self.velocity = self.speed * world.simulation.dt * Vector((np.cos(self.psi), np.sin(self.psi))) + self.wind

    def update_position(self):
        if self.intercepted is False:
            self.update_phi()
            self.update_psi()
            self.update_velocity()

            self.position = self.position + self.velocity

            self.x_positions.append(self.position.x)
            self.y_positions.append(self.position.y)

        if world.simulation.animate is True:
            self.plot.set_offsets((self.position.x, self.position.y))
            self.path.set_xdata(self.x_positions)
            self.path.set_ydata(self.y_positions)


class Prey:
    register = []

    def __init__(self,
                 initial_position = Vector((0,0)),
                 initial_psi = 0,  # np.random.uniform(0, 2*np.pi),
                 speed = 0.2,
                 trajectory = 'line',
                 omega = 0.1,
                 wind_speed = 0.0,
                 wind_direction = Vector((1,-1)),
                 colour = 'b'
                 ):

        self.position = initial_position
        self.psi = initial_psi
        self.speed = speed
        self.trajectory = trajectory
        self.omega = omega
        self.colour = colour

        self.x_positions = [self.position.x]
        self.y_positions = [self.position.y]

        self.velocity = self.speed * world.simulation.dt *Vector((np.cos(self.psi), np.sin(self.psi)))

        self.wind_speed = wind_speed
        self.wind_direction = wind_direction.normalized()

        if world.simulation.animate is True:
            self.plot = plt.scatter(self.position.y, self.position.y, c=self.colour)
            self.path, = plt.plot(self.x_positions, self.y_positions, lw=0.5, c=self.colour)

        Prey.register.append(self)

    def update_velocity(self, t):
        if self.trajectory == 'line':
            pass
        elif self.trajectory == 'sine':
            self.velocity += Vector((np.cos(self.omega * t), np.sin(self.omega * t)))
        elif self.trajectory == 'circle':
            self.velocity += self.omega * Matrix(([0, -1], [1, 0])) * self.velocity
        elif self.trajectory == 'random':
            self.velocity += np.random.normal(0, self.omega) * Vector((np.random.random()-0.5, np.random.random()-0.5)).normalized()

        self.velocity = self.speed * world.simulation.dt * self.velocity.normalized()

    def update_position(self, t):
        self.update_velocity(t)

        self.position += self.velocity  # + self.wind_speed * self.wind_direction

        self.x_positions.append(self.position.x)
        self.y_positions.append(self.position.y)

        if world.simulation.animate is True:
            self.path.set_xdata(self.x_positions)
            self.path.set_ydata(self.y_positions)
            self.plot.set_offsets((self.position.x, self.position.y))


class Simulator:
    def __init__(self, alphas, betas, wind_speed, wind_direction=Vector((0,0)), animate=False, start_y=50, omega=0., prey_speed=0.2):
        self.t = 0
        self.dt = 0.5

        self.animate = animate
        if animate is True:
            self.fig, self.ax = plt.subplots()

            plt.sca(self.ax)
            plt.xlim(-5, 60)
            plt.ylim(-5, start_y + 10)
            plt.xlabel('x (arb. units)')
            plt.ylabel('y (arb. units)')
            # plt.axis('equal')

        self.alphas = alphas
        self.betas = betas
        self.start_y = start_y
        self.omega = omega
        self.prey_speed = prey_speed

        self.times = {'direct': np.zeros((len(self.betas['direct']), len(self.alphas))),
                      'bearing': np.zeros((len(self.betas['bearing']), len(self.alphas)))}

        self.wind_speed = wind_speed
        self.wind_direction = wind_direction.normalized()

        self.finished = False

    def run(self):
        prey = Prey(Vector((0, self.start_y)), trajectory='random', omega=self.omega, speed=self.prey_speed, colour='k',
                    wind_direction=self.wind_direction, wind_speed=self.wind_speed)

        for alpha, beta in [(alpha, beta) for alpha in self.alphas for beta in self.betas['bearing']]:
            Predator(target=prey, pursuit_mode='bearing', colour='r', alpha=alpha, beta=beta, speed=0.3,
                     wind_direction=self.wind_direction, wind_speed=self.wind_speed)
        for alpha, beta in [(alpha, beta) for alpha in self.alphas for beta in self.betas['direct']]:
            Predator(target=prey, pursuit_mode='direct', colour='g', alpha=alpha, beta=beta, speed=0.3,
                     wind_direction=self.wind_direction, wind_speed=self.wind_speed)


        if self.animate is False:
            t = 0
            while self.finished is False:
                self.update_plots(t)
                t += 1
        else:
            ani = FuncAnimation(self.fig, self.update_plots, interval=1)

            plt.show(ani)

    def update_plots(self, t):
        self.t = t

        for prey in Prey.register:
            prey.update_position(t)

        for predator in Predator.register:
            predator.update_position()

        if Predator.done == len(Predator.register):
            self.analyse()

    def analyse(self):
        if False:# self.animate is True:
            fig, ax = plt.subplots(1,2)
            ax[0].grid('False')
            ax[1].grid('False')
            # print(1/self.times['direct'], '\n\n', 1/self.times['bearing'])
            ax[0].imshow(self.times['direct'], origin='lower')
            ax[1].imshow(self.times['bearing'], origin='lower')
            plt.show()

        self.finished = True
        if self.animate is True:
            plt.pause(100000)
            world.simulation.run_simulation()
            # self.animate = False
            # plt.close()



class World:
    def __init__(self, omega=0.):
        self.alphas = np.linspace(0.2, 1, 10)
        self.betas = {'direct': np.linspace(0.2, 1, 10),
                      'bearing': np.linspace(1, 2, 10)}

        self.wind_speed = 0.09
        self.wind_direction = Vector((0,0))
        self.simulation = None

        self.times = {'direct': np.zeros((len(self.betas['direct']), len(self.alphas))),
                      'bearing': np.zeros((len(self.betas['bearing']), len(self.alphas)))}

        self.start_y = 20
        self.omega = omega
        self.animate = True
        self.prey_speed = 0.2


    def run_simulation(self):
        self.simulation = Simulator(self.alphas, self.betas, self.wind_speed, Vector((np.random.rand() - 0.5, np.random.rand() - 0.5)), self.animate, self.start_y, self.omega, self.prey_speed)
        self.simulation.run()
        self.times['direct'] += self.simulation.times['direct']
        self.times['bearing'] += self.simulation.times['bearing']

    def run(self, n):
        for _ in tqdm(range(n)):
            self.run_simulation()
        # print(self.times)

        fig, axes = plt.subplots(1, 3)

        axes[0].imshow(self.times['direct'], origin='lower')
        axes[1].imshow(self.times['bearing'], origin='lower')

        xlabels = [f'{x:.2f}' for x in self.alphas]
        ylabels = {'direct': [f'{x:.2f}' for x in self.betas['direct']],
                   'bearing': [f'{x:.2f}' for x in self.betas['bearing']]}

        for ax in axes[0:2]:
            plt.sca(ax)
            plt.xticks(rotation=90)
            ax.set_xlabel('alpha')
            ax.set_ylabel('beta')
            ax.set_xticks(np.arange(0, len(self.alphas), 1))
            ax.set_xticklabels(xlabels)
            ax.grid('False')

        axes[0].set_title('Direct')
        axes[1].set_title('Bearing')
        axes[0].set_yticks(np.arange(0, len(self.betas['direct']), 1))
        axes[0].set_yticklabels(ylabels['direct'])
        axes[1].set_yticks(np.arange(0, len(self.betas['bearing']), 1))
        axes[1].set_yticklabels(ylabels['bearing'])

        axes[2].set_title('Comparison')
        axes[2].violinplot(self.times['direct'].flatten() / n, [0])
        axes[2].violinplot(self.times['bearing'].flatten() / n, [1])
        axes[2].set_xticks([0,1])
        axes[2].set_xticklabels(['direct', 'bearing'])
        axes[2].set_ylabel('time at interception')

        # plt.show()


world = World(0.)
world.run(100)
world = World(0.02)
world.run(100)
world = World(0.05)
world.run(100)
world = World(0.1)
world.run(100)
plt.show()
